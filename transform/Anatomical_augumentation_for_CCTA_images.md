# Anatomy-Informed Data Augmentation for Coronary Artery in CCTA Image
* Video of the on-the-fly anatomy-based data augmentation （slice view）
![image](https://github.com/xxsxxsxxs666/SparrowLink/assets/61532031/fd2e33e7-8a14-4f84-9bcc-4b5b9f9dd7fa.gif)
* Video of the on-the-fly anatomy-based data augmentation. （3D）
![image](https://github.com/xxsxxsxxs666/SparrowLink/assets/61532031/c2235188-02e9-427f-8f07-6fe7f807e540.gif)

<<<<<<< HEAD
* You can easily use our anatomy-based data augmentation tool by simply plug it into MONAI transform architecture:


   
=======
* You can use our anatomy-based data augmentation tool by simply plugging it into MONAI transform architecture:

```python
save_transform = Compose(
        [
            LoadImaged(keys=["image", "label", "heart"]),
            EnsureChannelFirstd(keys=["image", "label", "heart"]),
            ArteryTransformD(keys=["image", "label"], image_key="image", artery_key="label", p_anatomy_per_sample=1,
                             p_contrast_per_sample=1,
                             contrast_reduction_factor_range=(0.6, 1), mask_blur_range=(3, 6),
                             mvf_scale_factor_range=(1, 2), mode=("bilinear", "nearest")),
            # HeartTransformD(keys=["image", "label", "heart"], artery_key="label", heart_key="heart",
            #                 p_anatomy_heart=0, p_anatomy_artery=1,
            #                 dil_ranges=((-10, 10), (-5, -3)), directions_of_trans=((1, 1, 1), (1, 1, 1)), blur=(32, 8),
            #                 mode=("bilinear", "nearest", "nearest"), visualize=True, batch_interpolate=True,
            #                 threshold=(-1, 0.5, 0.5)),
            # CASTransformD(keys=["image", "label", "heart"], label_key="label", heart_key="heart", p_anatomy_per_sample=1,
            #               dil_ranges=((-30, -40), (-300, -500)), directions_of_trans=((1, 1, 1), (1, 1, 1)), blur=[4, 32],
            #               mode=("bilinear", "nearest", "nearest"),),
            SaveImaged(keys=["image"], output_dir=save_dir, output_postfix='spatial_transform_image',
                       print_log=True, padding_mode="zeros"),
            SaveImaged(keys=["label"], output_dir=save_dir, output_postfix='spatial_transform_label',
                       print_log=True, padding_mode="zeros"),
        ]
    )
```
>>>>>>> 53f3f0b0177b11eff7bc6e6db268fe7922e068ef
