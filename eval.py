
def temporal_IoU(timestamp_gt, timestamp_pred):
    union = (min(timestamp_gt[0], timestamp_pred[0]),
             max(timestamp_gt[1], timestamp_pred[1]))
    inter = (max(timestamp_gt[0], timestamp_pred[0]),
             min(timestamp_gt[1], timestamp_pred[1]))
    t_iou = 1.0*(inter[1]-inter[0])/(union[1]-union[0])
    if t_iou < 0:
        t_iou = 0
    return t_iou


def temporal_IoU_numpy(timestamp_gt, timestamp_pred):
    union = (min(timestamp_gt[0].numpy(), timestamp_pred[0].numpy()),
             max(timestamp_gt[1].numpy(), timestamp_pred[1].numpy()))
    inter = (max(timestamp_gt[0].numpy(), timestamp_pred[0].numpy()),
             min(timestamp_gt[1].numpy(), timestamp_pred[1].numpy()))
    t_iou = 1.0*(inter[1]-inter[0])/(union[1]-union[0])
    if t_iou < 0:
        t_iou = 0
    return t_iou
