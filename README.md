# DELandSSN
利用深度学习，进行超像素的生成，进而进行图像分割
1.	利用SSN和DEL完成图像分割的任务，并根据SSN和DEL设计了下面的这个网络。
2.	因为SSN和DEL前面部分的卷积网络是为了提取图像特征，于是我想用DEL的卷积网络替换SSN中的卷积，因为根据经验DEL的中卷积网络整合前几层的信息，包含的信息应该更完整。
3.	然后利用提取到的信息，一方面根据SSN后面的迭代过程，进行Superpixel的生成。另一方面利用Superpixel Pooling生成Superpixel Feature Vectors。
4.	最后根据生成的Superpixel和Superpixel Feature Vectors进行Merge,生成分割好的图像。
