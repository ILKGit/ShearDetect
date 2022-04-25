## Import packages
import os,sys, json, cv2, numpy as np, argparse, time

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


##--------------------------------------
## Data Augementation
##--------------------------------------
def train_transform():
    return A.Compose([
        A.Sequential([
            A.RandomRotate90(p=1), # Random rotation of an image by 90 degrees zero or more times
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, brightness_by_max=True, always_apply=False, p=1),# Random change of brightness & contrast
            A.HorizontalFlip(p=1), # Random perspective transform on an image
        ], p=1)
    ],
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bboxes_labels']) # Bboxes should have labels, read more at https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
    )


def main():
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model', default='model-1', help='DIR for output')
    parser.add_argument('--data', default='dataset', help='DIR of data. For creation of datasets see documenation')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--save_period', type=int, default=1, help='intervall for checkpoint saving')
    parser.add_argument('--weights', default=None, help='save weights dictonary *.pth-file.')
    ## get arguments
    args = parser.parse_args()
    ## define model name
    model_name  = args.model
    num_epochs  = args.epochs
    data_dir    = args.data
    save_period = args.save_period
    weights     = args.weights
    ## 
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    cpu_device = torch.device("cpu")

    print("device available: %s" %device)
    torch.cuda.empty_cache()

    ## define number of classes
    CLASSES = ['background','specimen','defect']
    num_classes = len(CLASSES)
    
    ## build output directories
    out_dir = os.path.join('model',model_name)
    ## check if model exists
    if os.path.exists(out_dir):
        sys.exit('%s exists\nchoose another name'%model_name)
    else:
        print('creating dir %s'%out_dir)
        utils.mkdir(out_dir)
        print('creating dir %s'%os.path.join(out_dir,'checkpoints'))
        utils.mkdir(os.path.join(out_dir,'checkpoints'))
        print('creating dir %s'%os.path.join(out_dir,'metrics'))
        utils.mkdir(os.path.join(out_dir,'metrics'))
        print('creating dir %s'%os.path.join(out_dir,'metrics'))
        utils.mkdir(os.path.join(out_dir,'results'))


    ## paths to datasets
    DIR_TRAIN  = os.path.join(data_dir,'train')
    DIR_VAL    = os.path.join(data_dir,'validation')
    DIR_TEST   = os.path.join(data_dir,'test')

    ## load datasets
    print("loading data")
    dataset_train = ClassDataset(DIR_TRAIN,CLASSES, 
                            transform=train_transform(),demo=False)
    dataset_val = ClassDataset(DIR_VAL,CLASSES, 
                            transform=None, demo=False)
    dataset_test = ClassDataset(DIR_TEST,CLASSES, 
                            transform=None, demo=False)

    data_loader_train = DataLoader(dataset_train, batch_size=20, shuffle=True, collate_fn=collate_fn)

    data_loader_val = DataLoader(dataset_val, batch_size=10, shuffle=False, collate_fn=collate_fn)

    data_loader_test = DataLoader(dataset_test, batch_size=30, shuffle=False, collate_fn=collate_fn)
    
    print("loading model")
    model = get_model(num_classes,
                    weights_path=weights
                    )
    model.to(device);


    ## model parameters
    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    optimizer = torch.optim.Adam(params, lr=0.001)
   
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                    T_max=33, eta_min=0.001, last_epoch=- 1, verbose=False)

    
    metrics_train = []
    metrics_val = []
    metrics_eval_test = []
    metrics_eval_val = []

    ## start training
    print("start traing of %s epoch(s)" %num_epochs)
    for epoch in range(num_epochs):
        metric_train, metric_val = train_one_epoch(model, optimizer,lr_scheduler,
                                                    data_loader_train,data_loader_val, 
                                                    device, epoch+1, print_freq=100)
        metrics_train.append(metric_train)
        metrics_val.append(metric_val)

        # save checkpoint every 'save_period' epoch
        if (epoch%save_period == save_period-1):
            ## evaluate model on validation and test set
            metrics_eval_test.append(evaluate(model, data_loader_test, device))
            metrics_eval_val.append(evaluate(model, data_loader_val, device))
            print('saving checkpoint from epoch %s' %(epoch+1))
            torch.save(model.state_dict(), 
                os.path.join(out_dir,'checkpoints', model_name + '_epoch-' + str(epoch+1) + '.pth'))

    ## Save model weights after training
    torch.save(model.state_dict(), os.path.join(out_dir, model_name + '.pth'))
    
    ## Store Training tracking in json
    list_training=[ 'lr','loss','loss_classifier',  
                    'loss_box_reg', 'loss_objectness',  
                    'loss_rpn_box_reg'] # 'time',  'data' ,'max mem'
    results_train = {}
    results_val = {}
    
    ## save data 
    for list_str in list_training:
        results_train[list_str] = []
        results_val[list_str] = []
        for epoch , (metric_train,metric_val) in enumerate(zip(metrics_train,metrics_val)):

            if list_str == 'lr':
                results_train[list_str].append(metric_train.meters[list_str].median)
            else:
                results_train[list_str].append([metric_train.meters[list_str].median,metric_train.meters[list_str].std])
                results_val[list_str].append([metric_val.meters[list_str].median,metric_val.meters[list_str].std])

    with open(os.path.join(out_dir,'metrics',model_name + '_results-train.json'), 'w') as outfile:
        json.dump(results_train, outfile)

    with open(os.path.join(out_dir,'metrics', model_name + '_results-val.json'), 'w') as outfile:
        json.dump(results_val, outfile)

    from utilities.utils import visualize_pred
    print('making ')

    with torch.no_grad():
        model.eval()
        for images, targets in data_loader_test:
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
                
                visualize_pred(image, bboxes,scores,labels,CLASSES,out_dir, bboxes_true,labels_true)



if __name__ == "__main__":
    main()