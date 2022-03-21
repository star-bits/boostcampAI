# boostcampAI

:duck: 부스트캠프 AI Tech 학습 내용 정리 (3기, CV)

---

Week 1

- [python stuff](https://github.com/star-bits/boostcampAI/blob/main/W1/%EC%A0%95%EB%A6%AC_python_stuff.ipynb): list comprehension, lambda, map, asterisk stuff(variable-length arguments, kwargs, unpacking), OOP, read(), pickle, csv, html parsing, xml, json
- numpy and pandas
- GD, Probability, Inference
- 심화 과제 1 정리: GD
- 심화 과제 2 정리: Backprop
- 심화 과제 3 정리: Maximum Liklihood Estimation

---

Week 2

- [AutoGrad stuff](https://github.com/star-bits/boostcampAI/blob/main/W2/%EC%A0%95%EB%A6%AC_PyTorch_AutoGrad.ipynb): 일반 방정식과 cost function이 들어간 forward propagation의 차이, or lack thereof; Linear regression에서 J와 J의 미분, 그리고 chain rule. ⭐
- [PyTorch axis](https://github.com/star-bits/boostcampAI/blob/main/W2/%E1%84%8C%E1%85%A5%E1%86%BC%E1%84%85%E1%85%B5_numpy_PyTorch_axis.ipynb): on numpy and PyTorch axis. TL;DR: axis 0 is always the 'most layered' axis - t.shape: torch.Size([(axis 0), (axis 1), (axis 2)]) ⭐
- [기본 과제 1 정리](https://github.com/star-bits/boostcampAI/blob/main/W2/%E1%84%8C%E1%85%A5%E1%86%BC%E1%84%85%E1%85%B5_%E1%84%80%E1%85%B5%E1%84%87%E1%85%A9%E1%86%AB1_Custom_Model.ipynb): PyTorch Function, Module, Model
- [기본 과제 2 정리](https://github.com/star-bits/boostcampAI/blob/main/W2/%E1%84%8C%E1%85%A5%E1%86%BC%E1%84%85%E1%85%B5_%EA%B8%B0%EB%B3%B82_Custom_Dataset_%26_Custom_DataLoader.ipynb): PyTorch Dataset, DataLoader
- [심화 과제 1 정리](https://github.com/star-bits/boostcampAI/blob/main/W2/%E1%84%8C%E1%85%A5%E1%86%BC%E1%84%85%E1%85%B5_%E1%84%89%E1%85%B5%E1%86%B7%E1%84%92%E1%85%AA1_Transfer_Learning_%26_Hyperparameter_Tuning.ipynb): loading pretrained model, modifying the number of output features of a layer, transforming dataset - Grayscale and ToTensor, hyperparameters, train and test ⭐
- [iterable (object) and iterator](https://github.com/star-bits/boostcampAI/blob/main/W2/%E1%84%8C%E1%85%A5%E1%86%BC%E1%84%85%E1%85%B5_iterable_iterator.ipynb)
- [generator](https://github.com/star-bits/boostcampAI/blob/main/W2/%EC%A0%95%EB%A6%AC_generator.ipynb)

---

Week 3

- [matplotlib intro](https://github.com/star-bits/boostcampAI/blob/main/W3/%EC%A0%95%EB%A6%AC_matplotlib.ipynb): fig, ax = plt.subplots(m, n); ax[i].plot(x, y); plt.show()
- [mpl: bar, line, scatter](https://github.com/star-bits/boostcampAI/blob/main/W3/%EC%A0%95%EB%A6%AC_mpl_bar_line_scatter.ipynb)
- [mpl: text, color, facet, misc.](https://github.com/star-bits/boostcampAI/blob/main/W3/%EC%A0%95%EB%A6%AC_mpl_text_color_facet_misc.ipynb)
- [mpl: seaborn](https://github.com/star-bits/boostcampAI/blob/main/W3/%E1%84%8C%E1%85%A5%E1%86%BC%E1%84%85%E1%85%B5_mpl_seaborn.ipynb)
- [mpl: polar, pie](https://github.com/star-bits/boostcampAI/blob/main/W3/%EC%A0%95%EB%A6%AC_mpl_polar_pie.ipynb)
- [mpl: missingno, squarify, pywaffle, matplotlib_venn](https://github.com/star-bits/boostcampAI/blob/main/W3/%EC%A0%95%EB%A6%AC_mpl_missing_treemap_waffle_venn.ipynb): missing data, treemap (e.g. finviz), waffle chart, venn diagram
- plotly express
- mpl: custom theme
- mpl: visualization techniques

---

Week 4

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

---

Week 5

- [shell commands](https://github.com/star-bits/boostcampAI/blob/main/W5/%EC%A0%95%EB%A6%AC_shell_commands.ipynb)
- [venv, conda](https://github.com/star-bits/boostcampAI/blob/main/W5/%EC%A0%95%EB%A6%AC_venv_conda.ipynb)
- [os cwd](https://github.com/star-bits/boostcampAI/blob/main/W5/%E1%84%8C%E1%85%A5%E1%86%BC%E1%84%85%E1%85%B5_os_cwd.ipynb)
- [docker](https://github.com/star-bits/boostcampAI/blob/main/W5/%EC%A0%95%EB%A6%AC_docker.ipynb)
- [mlflow](https://github.com/star-bits/boostcampAI/blob/main/W5/%EC%A0%95%EB%A6%AC_mlflow.ipynb)

---

Week 6-7

- [미션 2 정리](https://github.com/star-bits/boostcampAI/blob/main/W6-7/%EC%A0%95%EB%A6%AC_%EB%AF%B8%EC%85%982_EDA.ipynb): EDA ⭐
- [미션 3 정리](https://github.com/star-bits/boostcampAI/blob/main/W6-7/%EC%A0%95%EB%A6%AC_%EB%AF%B8%EC%85%983_Augmentation.ipynb): Augmentation ⭐
- [미션 4 정리](https://github.com/star-bits/boostcampAI/blob/main/W6-7/%EC%A0%95%EB%A6%AC_%EB%AF%B8%EC%85%984_Data_Generation.ipynb): Data Generation ⭐
- [미션 5 정리](https://github.com/star-bits/boostcampAI/blob/main/W6-7/%EC%A0%95%EB%A6%AC_%EB%AF%B8%EC%85%985_Model.ipynb): Model ⭐
- [미션 6 정리](https://github.com/star-bits/boostcampAI/blob/main/W6-7/%EC%A0%95%EB%A6%AC_%EB%AF%B8%EC%85%986_Pretrained.ipynb): Pretrained model ⭐
- [미션 7-8 정리](https://github.com/star-bits/boostcampAI/blob/main/W6-7/%E1%84%8C%E1%85%A5%E1%86%BC%E1%84%85%E1%85%B5_%E1%84%86%E1%85%B5%E1%84%89%E1%85%A7%E1%86%AB7-8_Loss_Optimizer.ipynb): Loss, Optimizer
- 미션 9 정리: Ensemble
- 미션 10 정리: tensorboard, wandb

---

Week 8-9

- [기본 과제 1 정리](https://github.com/star-bits/boostcampAI/blob/main/W8-9/%E1%84%8C%E1%85%A5%E1%86%BC%E1%84%85%E1%85%B5_resnet34_Implementation.ipynb): resnet34 implementation from scratch: ConvBlock(nn.Sequential(\*layers[nn.Conv2d, nn.BatchNorm2d, nn.ReLU])) -> ResBlock(nn.Sequential(\*layers[ConvBlock, ConvBlock, residual])) -> ResNet ⭐
- [기본 과제 2 정리](https://github.com/star-bits/boostcampAI/blob/main/W8-9/%E1%84%8C%E1%85%A5%E1%86%BC%E1%84%85%E1%85%B5_Data_Augmentation.ipynb): Data Augmentation - transforms.Compose([RandomCrop, ToTensor, Resize, Normalize]), Channel order: {cv2: BGR, torch: RGB}, Dimension: {cv2: (height, width, channel), torch conv2d layer: (batch_size, channel, height, width)} ⭐
- [기본 과제 3 정리](https://github.com/star-bits/boostcampAI/blob/main/W8-9/%E1%84%8C%E1%85%A5%E1%86%BC%E1%84%85%E1%85%B5_vgg11_Segmentation.ipynb): vgg11 implementation from scratch, semantic segmentation using vgg11 modified as FCN by replacing fc layer with 1x1 conv layer
- [심화 과제 1 정리](https://github.com/star-bits/boostcampAI/blob/main/W8-9/%EC%A0%95%EB%A6%AC_CNN_Visualization.ipynb): visualizing conv1 filters, visualizing model activations using forward hook, visualizing saliency map (gradient_logit/gradient_image), visualizing Grad-CAM ⭐
- [기본 과제 4 정리](https://github.com/star-bits/boostcampAI/blob/main/W8-9/%EC%A0%95%EB%A6%AC_CGAN.ipynb): CGAN - G(concat(emb(z), emb(y))), D(concat(emb(x), emb(y))) ⭐
- [기본 과제 5 정리](https://github.com/star-bits/boostcampAI/blob/main/W8-9/%E1%84%8C%E1%85%A5%E1%86%BC%E1%84%85%E1%85%B5_Multi-modal.ipynb): Multi-modal
- 심화 과제 2 정리
- 심화 과제 3 정리
- [More AutoGrad stuff](https://github.com/star-bits/boostcampAI/blob/main/W8-9/%EC%A0%95%EB%A6%AC_More_AutoGrad.ipynb)

---

Week 10-12

- 기본 미션 1 정리
- 심화 미션 1 정리
- 기본 미션 2 정리
- 기본 미션 4 정리
- 심화 미션 4 정리
- 기본 미션 5 정리
- 심화 미션 5 정리
