import os,sys, json, cv2, numpy as np, argparse, time

def main():

    parser = argparse.ArgumentParser(description='ShearDetect')
    parser.add_argument('--model', default='ShearDetect_model_v1.pth', help='Path to checkpoint / model')
    parser.add_argument('--data', default='dataset', help='DIR of data. For creation of datasets see documenation')
    parser.add_argument('--pred', default='pred', help='DIR for results')
    ## Import packages
    args = parser.parse_args()
    # print(args.accumulate(args.integers))   

    import torch
    from torch.utils.data import Dataset, DataLoader
    import torchvision

    print(torch.__version__)
    print(torchvision.__version__)

    import albumentations as A # Library for augmentations
    ## get custom functions
    # https://github.com/pytorch/vision/tree/main/references/detection
    import utilities.transforms as transforms, utilities.utils as utils, utilities.engine as engine
    from utilities.utils import collate_fn, ClassDataset, get_model
    from utilities.engine import evaluate, train_one_epoch

    ## define model name
    weights     = args.model
    data_dir    = args.data
    pred_dir    = args.pred

    ## 
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    cpu_device = torch.device("cpu")

    print("device available: %s" %device)
    torch.cuda.empty_cache()

    ## define number of classes
    CLASSES = ['background','specimen','defect']
    num_classes = len(CLASSES)


    ## LOAD the Model with checpoint
    model = get_model(num_classes,
                    weights_path=weights
                    )
    model.to(device);


    DIR_PRED   = os.path.join(data_dir)

    ## load datasets
    print("loading data")
    dataset_pred = ClassDataset(DIR_PRED,CLASSES)

    data_loader_pred = DataLoader(dataset_pred, batch_size=10, shuffle=False, collate_fn=collate_fn)


    from utilities.utils import visualize_pred
    print('making ')

    with torch.no_grad():
        model.eval()
        for images, targets in data_loader_pred:
            images = list(img.to(device) for img in images)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model_time = time.time()
            outputs = model(images)
            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            model_time = time.time() - model_time

            for image, out,target in zip(images,outputs,targets):
                scores  = out['scores'].detach().cpu().numpy()
                image = (image.permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)
    
                high_scores_idxs = np.where(scores > 0.5)[0].tolist() 
                # Indexes of boxes left after applying NMS (iou_threshold=0.3)
                post_nms_idxs = torchvision.ops.nms(out['boxes'][high_scores_idxs], out['scores'][high_scores_idxs], 0.3).cpu().numpy() 
                
                bboxes = []
                scores = []
                labels = []
                for bbox,score,label in zip(out['boxes'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy(),out['scores'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy(),out['labels'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy()):
                    bboxes.append(list(map(int, bbox.tolist())))
                    scores.append(score)
                    labels.append(label)
        
                bboxes_true = target['boxes'].detach().cpu().numpy().astype(np.int32).tolist()
                labels_true = target['labels'].detach().cpu().numpy().astype(np.int32).tolist()
                
                visualize_pred(image, bboxes,scores,labels,CLASSES,pred_dir)#, bboxes_true,labels_true)

if __name__ == "__main__":
    main()