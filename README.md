# boostcampAI

🦆 부스트캠프 AI Tech 학습 내용 정리 (3기, CV)

## Week 1

- [python stuff](https://github.com/star-bits/boostcampAI/blob/main/W1/%EC%A0%95%EB%A6%AC_python_stuff.ipynb): list comprehension, lambda, map, asterisk stuff(variable-length arguments, kwargs, unpacking), OOP, read(), pickle, csv, html parsing, xml, json
- [numpy and pandas](https://github.com/star-bits/boostcampAI/blob/main/W1/%EC%A0%95%EB%A6%AC_numpy_pandas.ipynb)
- GD, Probability, Inference
- 심화 과제 1 정리: GD
- 심화 과제 2 정리: Backprop
- 심화 과제 3 정리: Maximum Liklihood Estimation

## Week 2

- [AutoGrad stuff](https://github.com/star-bits/boostcampAI/blob/main/W2/%EC%A0%95%EB%A6%AC_PyTorch_AutoGrad.ipynb): 일반 방정식과 cost function이 들어간 forward propagation의 차이, or lack thereof; Linear regression에서 J와 J의 미분, 그리고 chain rule. ⭐
- [PyTorch axis](https://github.com/star-bits/boostcampAI/blob/main/W2/%E1%84%8C%E1%85%A5%E1%86%BC%E1%84%85%E1%85%B5_numpy_PyTorch_axis.ipynb): on numpy and PyTorch axis. TL;DR: axis 0 is always the 'most layered' axis - t.shape: torch.Size([(axis 0), (axis 1), (axis 2)]) ⭐
- [기본 과제 1 정리](https://github.com/star-bits/boostcampAI/blob/main/W2/%E1%84%8C%E1%85%A5%E1%86%BC%E1%84%85%E1%85%B5_%E1%84%80%E1%85%B5%E1%84%87%E1%85%A9%E1%86%AB1_Custom_Model.ipynb): PyTorch Function, Module, Model
- [기본 과제 2 정리](https://github.com/star-bits/boostcampAI/blob/main/W2/%E1%84%8C%E1%85%A5%E1%86%BC%E1%84%85%E1%85%B5_%EA%B8%B0%EB%B3%B82_Custom_Dataset_%26_Custom_DataLoader.ipynb): PyTorch Dataset, DataLoader
- [심화 과제 1 정리](https://github.com/star-bits/boostcampAI/blob/main/W2/%E1%84%8C%E1%85%A5%E1%86%BC%E1%84%85%E1%85%B5_%E1%84%89%E1%85%B5%E1%86%B7%E1%84%92%E1%85%AA1_Transfer_Learning_%26_Hyperparameter_Tuning.ipynb): loading pretrained model, modifying the number of output features of a layer, transforming dataset - Grayscale and ToTensor, hyperparameters, train and test ⭐
- [iterable (object) and iterator](https://github.com/star-bits/boostcampAI/blob/main/W2/%E1%84%8C%E1%85%A5%E1%86%BC%E1%84%85%E1%85%B5_iterable_iterator.ipynb)
- [generator](https://github.com/star-bits/boostcampAI/blob/main/W2/%EC%A0%95%EB%A6%AC_generator.ipynb)

## Week 3

- [matplotlib.pyplot as plt](https://github.com/star-bits/boostcampAI/blob/main/W3/%E1%84%8C%E1%85%A5%E1%86%BC%E1%84%85%E1%85%B5_matplotlib.ipynb): fig, ax = plt.subplots(m, n); ax[i].plot(x, y); plt.show()
- [mpl: bar, line, scatter](https://github.com/star-bits/boostcampAI/blob/main/W3/%EC%A0%95%EB%A6%AC_mpl_bar_line_scatter.ipynb)
- [mpl: text, color, facet, misc.](https://github.com/star-bits/boostcampAI/blob/main/W3/%EC%A0%95%EB%A6%AC_mpl_text_color_facet_misc.ipynb)
- [seaborn as sns](https://github.com/star-bits/boostcampAI/blob/main/W3/%E1%84%8C%E1%85%A5%E1%86%BC%E1%84%85%E1%85%B5_seaborn.ipynb)
- [mpl: polar, pie](https://github.com/star-bits/boostcampAI/blob/main/W3/%EC%A0%95%EB%A6%AC_mpl_polar_pie.ipynb)
- [mpl: missingno, squarify, pywaffle, matplotlib_venn](https://github.com/star-bits/boostcampAI/blob/main/W3/%EC%A0%95%EB%A6%AC_mpl_missing_treemap_waffle_venn.ipynb): missing data, treemap (e.g. finviz), waffle chart, venn diagram
- [plotly.express as px](https://github.com/star-bits/boostcampAI/blob/main/W3/%EC%A0%95%EB%A6%AC_plotly_express.ipynb)
- [mpl: custom theme](https://github.com/star-bits/boostcampAI/blob/main/W3/%EC%A0%95%EB%A6%AC_mpl_custom_theme.ipynb)
- visualization techniques

## Week 4

- [Optimization - Adam](https://github.com/star-bits/boostcampAI/blob/main/W4/%E1%84%8C%E1%85%A5%E1%86%BC%E1%84%85%E1%85%B5_Optimization_Adam.ipynb): cross validation (k-fold validation), bootstrapping/bagging/boosting, momentum - directions with intertia, RMSprop - adaptive learning rate, Adam, parameter norm penalty (weight decay) 
- [CNN](https://github.com/star-bits/boostcampAI/blob/main/W4/%EC%A0%95%EB%A6%AC_CNN.ipynb): AlexNet - ReLU solves the vanishing gradient problem, VGGNet - smaller kernel size (3x3), GoogLeNet - 1x1 convolution the channel-wise dimension reducer, ResNet - skip connection (addition), DenseNet - skip connection (concatenation) 
- [RNN](https://github.com/star-bits/boostcampAI/blob/main/W4/%E1%84%8C%E1%85%A5%E1%86%BC%E1%84%85%E1%85%B5_RNN.ipynb): vanishing/exploding gradient in RNN caused by sigmoid and ReLU, LSTM
- Transformer, ViT
- Generative model
- [기본 과제 1 정리](https://github.com/star-bits/boostcampAI/blob/main/W4/%E1%84%8C%E1%85%A5%E1%86%BC%E1%84%85%E1%85%B5_%E1%84%80%E1%85%B5%E1%84%87%E1%85%A9%E1%86%AB1_MLP.ipynb): MLP ⭐
- [기본 과제 2 정리](https://github.com/star-bits/boostcampAI/blob/main/W4/%EC%A0%95%EB%A6%AC_%EA%B8%B0%EB%B3%B82_Optimization.ipynb): Optimization
- [기본 과제 3 정리](https://github.com/star-bits/boostcampAI/blob/main/W4/%EC%A0%95%EB%A6%AC_%EA%B8%B0%EB%B3%B83_CNN.ipynb): CNN ⭐
- [기본 과제 4 정리](https://github.com/star-bits/boostcampAI/blob/main/W4/%EC%A0%95%EB%A6%AC_%EA%B8%B0%EB%B3%B84_LSTM.ipynb): LSTM
- [기본 과제 5 정리](https://github.com/star-bits/boostcampAI/blob/main/W4/%EC%A0%95%EB%A6%AC_%EA%B8%B0%EB%B3%B85_SDPA.ipynb): SDPA
- [심화 과제 1 정리](https://github.com/star-bits/boostcampAI/blob/main/W4/%EC%A0%95%EB%A6%AC_%EC%8B%AC%ED%99%941_ViT.ipynb): ViT
- [심화 과제 2 정리](https://github.com/star-bits/boostcampAI/blob/main/W4/%EC%A0%95%EB%A6%AC_%EC%8B%AC%ED%99%942_AAE.ipynb): AAE

## Week 5

- [shell commands](https://github.com/star-bits/boostcampAI/blob/main/W5/%EC%A0%95%EB%A6%AC_shell_commands.ipynb)
- [venv, conda](https://github.com/star-bits/boostcampAI/blob/main/W5/%EC%A0%95%EB%A6%AC_venv_conda.ipynb)
- [os cwd](https://github.com/star-bits/boostcampAI/blob/main/W5/%E1%84%8C%E1%85%A5%E1%86%BC%E1%84%85%E1%85%B5_os_cwd.ipynb)
- [docker](https://github.com/star-bits/boostcampAI/blob/main/W5/%EC%A0%95%EB%A6%AC_docker.ipynb)
- [mlflow](https://github.com/star-bits/boostcampAI/blob/main/W5/%EC%A0%95%EB%A6%AC_mlflow.ipynb)

## Week 6-7: Image Classification

- [미션 2 정리](https://github.com/star-bits/boostcampAI/blob/main/W6-7/%EC%A0%95%EB%A6%AC_%EB%AF%B8%EC%85%982_EDA.ipynb): EDA ⭐
- [미션 3 정리](https://github.com/star-bits/boostcampAI/blob/main/W6-7/%EC%A0%95%EB%A6%AC_%EB%AF%B8%EC%85%983_Augmentation.ipynb): Augmentation ⭐
- [미션 4 정리](https://github.com/star-bits/boostcampAI/blob/main/W6-7/%EC%A0%95%EB%A6%AC_%EB%AF%B8%EC%85%984_Data_Generation.ipynb): Data Generation ⭐
- [미션 5 정리](https://github.com/star-bits/boostcampAI/blob/main/W6-7/%EC%A0%95%EB%A6%AC_%EB%AF%B8%EC%85%985_Model.ipynb): Model ⭐
- [미션 6 정리](https://github.com/star-bits/boostcampAI/blob/main/W6-7/%EC%A0%95%EB%A6%AC_%EB%AF%B8%EC%85%986_Pretrained.ipynb): Pretrained model ⭐
- [미션 7-8 정리](https://github.com/star-bits/boostcampAI/blob/main/W6-7/%E1%84%8C%E1%85%A5%E1%86%BC%E1%84%85%E1%85%B5_%E1%84%86%E1%85%B5%E1%84%89%E1%85%A7%E1%86%AB7-8_Loss_Optimizer.ipynb): Loss, Optimizer
- 미션 9 정리: Ensemble
- 미션 10 정리: tensorboard, wandb

## Week 8-9

- [기본 과제 1 정리](https://github.com/star-bits/boostcampAI/blob/main/W8-9/%E1%84%8C%E1%85%A5%E1%86%BC%E1%84%85%E1%85%B5_resnet34_Implementation.ipynb): resnet34 implementation from scratch: ConvBlock(nn.Sequential(\*layers[nn.Conv2d, nn.BatchNorm2d, nn.ReLU])) -> ResBlock(nn.Sequential(\*layers[ConvBlock, ConvBlock, residual])) -> ResNet ⭐
- [기본 과제 2 정리](https://github.com/star-bits/boostcampAI/blob/main/W8-9/%E1%84%8C%E1%85%A5%E1%86%BC%E1%84%85%E1%85%B5_Data_Augmentation.ipynb): Data Augmentation - transforms.Compose([RandomCrop, ToTensor, Resize, Normalize]), Channel order: {cv2: BGR, torch: RGB}, Dimension: {cv2: (height, width, channel), torch conv2d layer: (batch_size, channel, height, width)} ⭐
- [기본 과제 3 정리](https://github.com/star-bits/boostcampAI/blob/main/W8-9/%E1%84%8C%E1%85%A5%E1%86%BC%E1%84%85%E1%85%B5_vgg11_Segmentation.ipynb): vgg11 implementation from scratch, semantic segmentation using vgg11 modified as FCN by replacing fc layer with 1x1 conv layer
- [심화 과제 1 정리](https://github.com/star-bits/boostcampAI/blob/main/W8-9/%EC%A0%95%EB%A6%AC_CNN_Visualization.ipynb): visualizing conv1 filters, visualizing model activations using forward hook, visualizing saliency map (gradient_logit/gradient_image), visualizing Grad-CAM ⭐
- [기본 과제 4 정리](https://github.com/star-bits/boostcampAI/blob/main/W8-9/%EC%A0%95%EB%A6%AC_CGAN.ipynb): CGAN - G(concat(emb(z), emb(y))), D(concat(emb(x), emb(y))) ⭐
- [기본 과제 5 정리](https://github.com/star-bits/boostcampAI/blob/main/W8-9/%E1%84%8C%E1%85%A5%E1%86%BC%E1%84%85%E1%85%B5_Multi-modal.ipynb): Multi-modal
- [심화 과제 2 정리](https://github.com/star-bits/boostcampAI/blob/main/W8-9/%EC%A0%95%EB%A6%AC_Hourglass_Network.ipynb): Hourglass network, torchsummary summary
- [심화 과제 3 정리](https://github.com/star-bits/boostcampAI/blob/main/W8-9/%EC%A0%95%EB%A6%AC_Depth_map.ipynb): Depth map
- [More AutoGrad stuff](https://github.com/star-bits/boostcampAI/blob/main/W8-9/%EC%A0%95%EB%A6%AC_More_AutoGrad.ipynb)

## Week 10-12: Object Detection

- [Two-Stage Detectors](https://github.com/star-bits/boostcampAI/blob/main/W10-12/%EC%A0%95%EB%A6%AC_2_Stage_Detectors.ipynb): R-CNN, SPPNet (ROI projection: projection of selective search result onto a feature map, Spatial Pyramid Pooling: n by n grid pooling - fixed fc layer size) solves multiple CNN problem and image warping problem, Fast R-CNN (multi-task loss: classification loss + bounding box regression loss), Faster R-CNN (Region Proposal Network: apply anchor boxes on feature map cells)
- [Feature Pyramid Network](https://github.com/star-bits/boostcampAI/blob/main/W10-12/%EC%A0%95%EB%A6%AC_Feature_Pyramid_Network.ipynb): FPN (top-down pathway: mixing low level and high level feature maps), PANet (bottom-up path augmentation, adaptive feature pooling: ROI from all stages), Recursive FPN, Bi-directional FPN, NAS(Neural Architecture Search)FPN 
- [One-Stage Detectors](https://github.com/star-bits/boostcampAI/blob/main/W10-12/%EC%A0%95%EB%A6%AC_1_Stage_Detectors.ipynb): YOLO (loss: localization loss + confidence loss + classification loss), SSD(multi-scale feature maps, no fc layer, has anchor box), RetinaNet(background class imbalance - solved by focal loss)
- [More on Two-Stage Detectors](https://github.com/star-bits/boostcampAI/blob/main/W10-12/%E1%84%8C%E1%85%A5%E1%86%BC%E1%84%85%E1%85%B5_More_2_Stage_Detectors.ipynb): Faster R-CNN (image -> through ConvNet -> feature map -> through RPN -> ROI; (ROI + feature map) -> through ROI pooling -> through (classification head + regressor head) -> output) has 3 networks: (ConvNet, RPN, cls+reg head), RPN: (9 anchor boxes, 0 or 1 classification, NMS), Cascade R-CNN, Deformable convolution, Transformer (Q, K, V created by W_Q, W_K, W_V; Attention map from Q, K), Swin ⭐
- [More on One-Stage Detectors](https://github.com/star-bits/boostcampAI/blob/main/W10-12/%EC%A0%95%EB%A6%AC_More_1_Stage_Detectors.ipynb): Two-stage detectors: prediction doesn't happen at every pixel (of a final feature map), proposals from RPN gets projected onto a feature map, and after going through ROI pooling, output gets delivered to cls head and reg head; One-stage detectors: prediction gets made from every pixel (of a final feature map), doesn't have RPN - detector itself is an alteration of RPN, each pixel gets anchor boxes and classification and bbox regression comes right after ⭐
- [기본 미션 1 정리](https://github.com/star-bits/boostcampAI/blob/main/W10-12/%E1%84%8C%E1%85%A5%E1%86%BC%E1%84%85%E1%85%B5_Metric.ipynb): bbox mAP
- 심화 미션 1 정리: bbox mAP (advanced)
- 기본 미션 2 정리: Faster R-CNN
- 기본 미션 4 정리: FPN
- 심화 미션 4 정리: Faster R-CNN FPN
- 기본 미션 5 정리: YOLO
- 심화 미션 5 정리: YOLO inference
- 심화 미션 6 정리: YOLOv3
- [기본 미션 7 정리](https://github.com/star-bits/boostcampAI/blob/main/W10-12/%EC%A0%95%EB%A6%AC_WBF_Ensemble.ipynb): WBF(Weighted Boxes Fusion) ensemble
- Faster R-CNN with Swin-L backbone: [config.py](https://github.com/star-bits/boostcampAI/blob/main/W10-12/_swin_faster_config.py), [train.ipynb](https://github.com/star-bits/boostcampAI/blob/main/W10-12/_swin_faster_train.ipynb), [infer.ipynb](https://github.com/star-bits/boostcampAI/blob/main/W10-12/_swin_faster_infer.ipynb) ⭐
- UniverseNet with Swin-L backbone: [config.py](https://github.com/star-bits/boostcampAI/blob/main/W10-12/_universe_config.py), [train.ipynb](https://github.com/star-bits/boostcampAI/blob/main/W10-12/_universe_train.ipynb), [infer.ipynb](https://github.com/star-bits/boostcampAI/blob/main/W10-12/_universe_infer.ipynb)

## Week 13-14: Text Detection

- EDA.ipynb
- [download_ICDAR17.sh](https://github.com/star-bits/boostcampAI/blob/main/W13-14/download_ICDAR17.sh)
- [add_tr_va.sh](https://github.com/star-bits/boostcampAI/blob/main/W13-14/add_tr_va.sh) to rename images
- [mlt2ufo_ICDAR17raw2LKJ.py](https://github.com/star-bits/boostcampAI/blob/main/W13-14/convert_mlt_ICDAR17raw2LKJ.py)
- [mlt2ufo_ICDAR19raw2LKJ.py](https://github.com/star-bits/boostcampAI/blob/main/W13-14/convert_mlt_ICDAR19raw2LKJ.py)
- [im_mode_test.py](https://github.com/star-bits/boostcampAI/blob/main/W13-14/im_mode_test.py)
- [resize_dataset.py](https://github.com/star-bits/boostcampAI/blob/main/W13-14/resize_dataset.py): os.makedirs(), json.load(), json.dump(), cv2.imread(), cv2.resize(), cv2.imwrite()
- [cvtPoly2Rect.ipynb](https://github.com/star-bits/boostcampAI/blob/main/W13-14/cvtPoly2Rect.ipynb): cv2.contourArea(), cv2.minAreaRect(), cv2.boxPoints()
- [dataset.py](https://github.com/star-bits/boostcampAI/blob/main/W13-14/dataset_multiple.py) to check the len of dataset
- [train.py](https://github.com/star-bits/boostcampAI/blob/main/W13-14/train_multiple.py) (lr_scheduler: MultiStepLR | CosineAnnealingLR | CosineAnnealingWarmUpRestarts)

## Week 15-17: Semantic Segmentation

- [Semantic Segmentation Pipeline](https://github.com/star-bits/boostcampAI/blob/main/W15-17/%EC%A0%95%EB%A6%AC_Segmentation_Baseline.ipynb) (torchvision.models.segmentation.fcn_resnet50)
- Semantic Segmentation Models: FCN-32s, FCN-16s, FCN-8s, DeconvNet, SegNet, DeepLabV1, DialatedNet, DeepLabV2-VGG16, DeepLabV2-ResNet101, DeepLabV3-ResNet101, DeepLabV3Plus-Xception, UNet, UNet++
- [UNet3Plus.py](https://github.com/star-bits/boostcampAI/blob/main/W15-17/UNet3Plus.py): UNet3+ from scratch
- [UNet3+ Implementation](https://github.com/star-bits/boostcampAI/blob/main/W15-17/%EC%A0%95%EB%A6%AC_Segmentation_UNet3Plus.ipynb) ⭐

## Week 18-22: Final Project

- [웹캠을 통한 가상 마우스](https://github.com/boostcampaitech3/final-project-level3-cv-13): 손과 웹캠으로 마우스 커서 조작하기
  - 웹캠으로 손의 키포인트를 찾고, 제스쳐를 인식해 마우스 커서를 조작하는 프로젝트.
  - 해당 태스크에 특화된 데이터셋을 만들기 위해 직접 다양한 환경에서 데이터를 만들어주는 코드와, 그 데이터의 COCO 포맷 어노테이션 파일을 만들어주는 코드를 작성.
  - MMPose에서 ImageNet pretrained MobileNetV3-Large를 이용해, 대규모 공개 데이터셋(FreiHAND)으로 학습하고, 그 모델을 다시 pretrained삼아 직접 제작한 데이터셋에 재학습하는 방식으로 핸드 키포인트 디텍션 정확도 및 안정성 향상.
