import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import l2norm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Embedder(nn.Module):
    '''idで示されている単語をベクトルに変換します'''

    def __init__(self, weights):
        super(Embedder, self).__init__()

        self.embeddings = nn.Embedding.from_pretrained(
            embeddings=weights, freeze=True)
        # freeze=Trueによりバックプロパゲーションで更新されず変化しなくなります

    def forward(self, x):
        x_vec = self.embeddings(x)

        return x_vec


class PositionalEncoder(nn.Module):
    '''入力された単語の位置を示すベクトル情報を付加する'''

    def __init__(self, d_model=300, max_seq_len=85):
        super().__init__()

        self.d_model = d_model  # 単語ベクトルの次元数

        # 単語の順番（pos）と埋め込みベクトルの次元の位置（i）によって一意に定まる値の表をpeとして作成
        pe = torch.zeros(max_seq_len, d_model)

        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos /
                                          (10000 ** ((2 * (i + 1))/d_model)))

        # 表peの先頭に、ミニバッチ次元となる次元を足す
        self.pe = pe.unsqueeze(0)

        # 勾配を計算しないようにする
        self.pe.requires_grad = False

    def forward(self, x):
        # GPUが使える場合はGPUへ送る
        self.pe = self.pe.to(device)

        # 入力xとPositonal Encodingを足し算する
        # xがpeよりも小さいので、大きくする
        ret = math.sqrt(self.d_model)*x + self.pe
        return ret


class Attention(nn.Module):
    '''Transformerは本当はマルチヘッドAttentionですが、
    分かりやすさを優先しシングルAttentionで実装します'''

    def __init__(self, d_model=300):
        super().__init__()

        # SAGANでは1dConvを使用したが、今回は全結合層で特徴量を変換する
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        # 出力時に使用する全結合層
        self.out = nn.Linear(d_model, d_model)

        # Attentionの大きさ調整の変数
        self.d_k = d_model

    def forward(self, q, k, v, mask):
        # 全結合層で特徴量を変換
        k = self.k_linear(k)
        q = self.q_linear(q)
        v = self.v_linear(v)

        # Attentionの値を計算する
        # 各値を足し算すると大きくなりすぎるので、root(d_k)で割って調整
        weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.d_k)

        # ここでmaskを計算
        mask = mask.unsqueeze(1)
        weights = weights.masked_fill(mask == 0, -1e9)

        # softmaxで規格化をする
        normlized_weights = F.softmax(weights, dim=-1)

        # AttentionをValueとかけ算
        output = torch.matmul(normlized_weights, v)

        # 全結合層で特徴量を変換
        output = self.out(output)

        return output, normlized_weights


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=1024, dropout=0.1):
        '''Attention層から出力を単純に全結合層2つで特徴量を変換するだけのユニットです'''
        super().__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.dropout(F.relu(x))
        x = self.linear_2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, attn_mode, max_seq_len, dropout=0.1):
        super().__init__()

        self.max_seq_len = max_seq_len

        # LayerNormalization層
        # https://pytorch.org/docs/stable/nn.html?highlight=layernorm
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)

        # Attention層
        self.attn_mode = attn_mode
        self.attn = Attention(d_model)
        self.multi_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=4)

        # Attentionのあとの全結合層2つ
        self.ff = FeedForward(d_model)

        # Dropout
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 正規化とAttention
        x_normlized = self.norm_1(x)
        if self.attn_mode == 'simple':
            output, normlized_weights = self.attn(
                x_normlized, x_normlized, x_normlized, mask)
        elif self.attn_mode == 'multihead':
            x_normlized = x_normlized.permute(1, 0, 2)
            output, normlized_weights = self.multi_attn(
                x_normlized, x_normlized, x_normlized)
            output = output.permute(1, 0, 2)

        embed = x + self.dropout_1(output)

        # 正規化と全結合層
        x_normlized2 = self.norm_2(embed)
        output = embed + self.dropout_2(self.ff(x_normlized2))

        return output, normlized_weights


class TransformerLinear(nn.Module):
    '''Transformer_Blockの出力を使用し、最後にクラス分類させる'''

    def __init__(self, d_model=300, output_dim=256):
        super().__init__()

        # 全結合層
        self.linear = nn.Linear(d_model, output_dim)

        # 重み初期化処理
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)

    def forward(self, x):
        # x0 = x[:, 0, :]  # 各ミニバッチの各文の先頭の単語の特徴量（300次元）を取り出す
        out = self.linear(x)  # (x0)

        return out


# 最終的なTransformerモデルのクラス


class TransformerClassification(nn.Module):
    '''Transformerでクラス分類させる'''

    def __init__(self, batch_size, weights, d_model=300, max_seq_len=85, max_verb_len=15, output_dim=256, attn_mode='multihead'):
        super().__init__()
        self.batch_size = batch_size
        self.max_len = max_seq_len + max_verb_len
        self.output_dim = output_dim

        # モデル構築
        self.sen_embedder = Embedder(weights)
        self.verb_embedder = Embedder(weights)
        self.sen_pos_enc = PositionalEncoder(
            d_model=d_model, max_seq_len=max_seq_len)
        self.verb_pos_enc = PositionalEncoder(
            d_model=d_model, max_seq_len=max_verb_len)
        self.cat_linear = nn.Linear(d_model, d_model)
        self.transformer_block_1 = TransformerBlock(
            d_model=d_model, attn_mode=attn_mode, max_seq_len=self.max_len)
        self.transformer_block_2 = TransformerBlock(
            d_model=d_model, attn_mode=attn_mode, max_seq_len=self.max_len)
        self.transformer_linear = TransformerLinear(
            output_dim=output_dim, d_model=d_model)
        self.linear = nn.Linear(self.max_len, 1)

    def forward(self, iter_num, batch_sentence, batch_verb, vocab):
        mask = (batch_sentence != 0)
        sen_embed = self.sen_embedder(batch_sentence)  # 単語をベクトルに
        verb_embed = self.verb_embedder(batch_verb)
        sen_pos_embed = self.sen_pos_enc(sen_embed)  # Positon情報を足し算
        verb_pos_embed = self.verb_pos_enc(verb_embed)
        pos_embed = torch.cat((sen_pos_embed, verb_pos_embed), dim=1)
        pos_embed = self.cat_linear(pos_embed)
        output_1, normlized_weights_1 = self.transformer_block_1(
            pos_embed, mask)  # Self-Attentionで特徴量を変換
        output_2, normlized_weights_2 = self.transformer_block_2(
            output_1, mask)  # Self-Attentionで特徴量を変換
        # print(output_1.size(), output_2.size())
        output = self.transformer_linear(
            output_2)  # 最終出力の0単語目を使用して、分類0-1のスカラーを出力
        sen2vec = self.linear(output.permute(0, 2, 1)).squeeze(dim=2)
        sen2vec = l2norm(sen2vec)
        output = output.repeat_interleave(
            4, dim=0).reshape(self.batch_size, self.max_len, -1)
        # print(sen2vec.size(), output.size())
        return sen2vec, output  # , normlized_weights_1, normlized_weights_2


class TransformerClassification1(nn.Module):
    '''Transformerでクラス分類させる'''

    def __init__(self, batch_size, weights, d_model=300, max_seq_len=85, output_dim=256, attn_mode='multihead'):
        super().__init__()
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.output_dim = output_dim

        # モデル構築
        self.embedder = Embedder(weights)
        self.pos_enc = PositionalEncoder(
            d_model=d_model, max_seq_len=max_seq_len)
        self.transformer_block_1 = TransformerBlock(
            d_model=d_model, attn_mode=attn_mode, max_seq_len=max_seq_len)
        self.transformer_block_2 = TransformerBlock(
            d_model=d_model, attn_mode=attn_mode, max_seq_len=max_seq_len)
        self.transformer_linear = TransformerLinear(
            output_dim=output_dim, d_model=d_model)
        self.linear = nn.Linear(max_seq_len, 1)

    def forward(self, iter_num, batch_sentence, vocab):
        mask = (batch_sentence != 0)
        embed = self.embedder(batch_sentence)  # 単語をベクトルに
        pos_embed = self.pos_enc(embed)  # Positon情報を足し算
        output_1, normlized_weights_1 = self.transformer_block_1(
            pos_embed, mask)  # Self-Attentionで特徴量を変換
        output_2, normlized_weights_2 = self.transformer_block_2(
            output_1, mask)  # Self-Attentionで特徴量を変換
        # print(output_1.size(), output_2.size())
        output = self.transformer_linear(
            output_2)  # 最終出力の0単語目を使用して、分類0-1のスカラーを出力
        sen2vec = self.linear(output.permute(0, 2, 1)).squeeze(dim=2)
        sen2vec = l2norm(sen2vec)
        output = output.repeat_interleave(
            4, dim=0).reshape(self.batch_size, self.max_seq_len, -1)
        # print(sen2vec.size(), output.size())
        return sen2vec, output  # , normlized_weights_1, normlized_weights_2
