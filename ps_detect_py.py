import cv2
import torch
import numpy as np
import math

from models.experimental import attempt_load

def main():
    video_path = '/opt_disk1/share/Dataset/parking_slot_lh/raw/videos/20220919/2019-10-31    01-02-44-环视.h264'
    Videocapture = cv2.VideoCapture(video_path)
    outPath = './out20220919-010244-nano3123.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(outPath, fourcc, 25.0, (1280, 960))

    device = torch.device('cuda:0')#('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torch.jit.load('./out_models/nano_st32_pic3123_torchscript_cuda0.pt', map_location=device).to(device)
    # model.model[-1].export = True

    model.eval()
    while Videocapture.isOpened():
        ret, frame = Videocapture.read()
        if ret:
            frame1 = frame[40:, 180:1100]
            frame1 = frame1[:,:,::-1]
            frame1 = cv2.resize(frame1, (640, 640))
            img = torch.from_numpy(frame1.transpose(2, 0, 1).astype(np.float32) / 255.0).to(device)
            img = img.unsqueeze(0)
            with torch.no_grad():
                pred = model(img)
            # print(len(pred))
            pred = pred[0]
            # pred = pred.sigmoid()
            pred = pred.cpu().numpy()
            # print(pred.shape)
            pred = pred[pred[:, 7] > 0.5]
            pred = sorted(pred, key=lambda pred: pred[7], reverse=True)
            valid = np.ones(len(pred), dtype=int)
            slot = []
            for i in range(len(pred)):
                if not valid[i]:
                    continue
                slot.append(pred[i])
                for j in range(i+1, len(pred)):
                    if not valid[j]:
                        continue
                    if math.dist(pred[i][0:2], pred[j][0:2]) < 40:
                        valid[j] = 0

            # Draw
            cls_list = [[0, 'Slot', (0, 255, 0)],
                        [1, 'Engaged', (0, 0, 255)]]
            for i in range(len(slot)):
                deltax = slot[i][6] * slot[i][2] / 2
                deltay = slot[i][6] * slot[i][3] / 2
                p1 = (int((slot[i][0] + deltax)*(920/640)+180), int((slot[i][1] + deltay)*(920/640)+40))
                p2 = (int((slot[i][0] - deltax)*(920/640)+180), int((slot[i][1] - deltay)*(920/640)+40))

                len2 = 40
                deltax = len2*slot[i][4]
                deltay = len2*slot[i][5]
                p3 = (int(p1[0]+deltax), int(p1[1]+deltay))
                p4 = (int(p2[0]+deltax), int(p2[1]+deltay))

                cls_id = np.argmax(slot[i][8:])
                cls_color = cls_list[cls_id][2]
                cv2.line(frame, p1, p2, cls_color, 2)
                cv2.line(frame, p1, p3, cls_color, 2)
                cv2.line(frame, p2, p4, cls_color, 2)

            out.write(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    Videocapture.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
