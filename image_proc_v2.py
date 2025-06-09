import random
from PIL import Image, ImageEnhance
import numpy as np
import cv2


def preprocess_label_multiclass(label, type='skin'):
    
    # Initial label prep assuming it is loaded as image
    label = np.array(label)
    label = 255 - label
    
    if type == 'skin':
        # apply threshold
        skin_threshold = 55
        label[(label <= (skin_threshold - 5))] = 0
        label[(label >= (skin_threshold + 5))] = 0
        label[(label > (skin_threshold - 5)) & (label < (skin_threshold + 5))] = 255
    elif type == 'body':
        # Apply full body threshold
        body_threshold = 20
        label[label > body_threshold] = 255
        label[label <= body_threshold] = 0
    elif type == 'clothes':
        # Apply clothes threshold
        clothes_threshold = 160
        label[(label <= (clothes_threshold - 5))] = 0
        label[(label >= (clothes_threshold + 5))] = 0
        label[(label > (clothes_threshold - 5)) & (label < (clothes_threshold + 5))] = 255
    elif type == 'multi':
        classes = [55, 160, 108, 85, 0] # skin, clothes, hair, object, background
        # For each label value write to a new class
        # 160 - clothes, 55 - skin, 108 - hair, 0 - background, 85 - object
        labels = np.zeros((*label.shape, 5), dtype=np.uint8)
        for i, color in enumerate(classes):
            lower_bound = max(0, color - 5)
            upper_bound = min(255, color + 5)
            labels[:, :, i] = np.where((label >= lower_bound) & (label <= upper_bound), 255, 0).astype(np.uint8)

        return labels
    else:
        label = 255 - label
        
    # Convert back to image
    label = Image.fromarray(label).convert('L')


    return label

def refine_foreground(image, mask, r=90):
    if mask.size != image.size:
        mask = mask.resize(image.size)
    image = np.array(image) / 255.0
    mask = np.array(mask) / 255.0
    estimated_foreground = FB_blur_fusion_foreground_estimator_2(image, mask, r=r)
    image_masked = Image.fromarray((estimated_foreground * 255.0).astype(np.uint8))
    return image_masked


def FB_blur_fusion_foreground_estimator_2(image, alpha, r=90):
    # Thanks to the source: https://github.com/Photoroom/fast-foreground-estimation
    alpha = alpha[:, :, None]
    F, blur_B = FB_blur_fusion_foreground_estimator(
        image, image, image, alpha, r)
    return FB_blur_fusion_foreground_estimator(image, F, blur_B, alpha, r=6)[0]


def FB_blur_fusion_foreground_estimator(image, F, B, alpha, r=90):
    if isinstance(image, Image.Image):
        image = np.array(image) / 255.0
    blurred_alpha = cv2.blur(alpha, (r, r))[:, :, None]

    blurred_FA = cv2.blur(F * alpha, (r, r))
    blurred_F = blurred_FA / (blurred_alpha + 1e-5)

    blurred_B1A = cv2.blur(B * (1 - alpha), (r, r))
    blurred_B = blurred_B1A / ((1 - blurred_alpha) + 1e-5)
    F = blurred_F + alpha * \
        (image - alpha * blurred_F - (1 - alpha) * blurred_B)
    F = np.clip(F, 0, 1)
    return F, blurred_B


def preproc(image, label, preproc_methods=['flip']):
    if 'flip' in preproc_methods:
        image, label = cv_random_flip(image, label)
    if 'crop' in preproc_methods:
        image, label = random_crop(image, label)
    if 'rotate' in preproc_methods:
        image, label = random_rotate(image, label)
    if 'enhance' in preproc_methods:
        image = color_enhance(image)
    if 'pepper' in preproc_methods:
        image = random_pepper(image)
    if 'blur' in preproc_methods:
        image = random_gaussian(image)
    if 'cutout' in preproc_methods:
        image, label = random_cutout(image, label)
    return image, label


def cv_random_flip(img, label):
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label


# def random_crop(image, label):
#     border = 30
#     image_width = image.size[0]
#     image_height = image.size[1]
#     border = int(min(image_width, image_height) * 0.1)
#     crop_win_width = np.random.randint(image_width - border, image_width)
#     crop_win_height = np.random.randint(image_height - border, image_height)
#     random_region = (
#         (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
#         (image_height + crop_win_height) >> 1)
#     return image.crop(random_region), label.crop(random_region)


def random_crop(image, label):

    if random.random() > 0.6:
        return image, label # Peace out
    
    # Determine if we should do mask-based crop instead (50% chance when we're cropping)
    do_mask_crop = random.random() > 0.5
    
    if do_mask_crop:
        # Set padding percentage to a random value between 0.02 and 0.10
        padding_pct = random.uniform(0.02, 0.10)
        # Convert label to numpy array to find mask boundaries
        label_np = np.array(label, dtype=np.uint8)  # Ensure correct dtype
        
        # Find the mask boundaries (where label > 0)
        if len(label_np.shape) == 3:  # Handle RGB masks
            label_np = label_np.mean(axis=2)  # Convert to grayscale if RGB
        rows = np.any(label_np > 0, axis=1)
        cols = np.any(label_np > 0, axis=0)
        
        # Check if mask is empty (no positive pixels)
        if not np.any(rows) or not np.any(cols):
            return image, label
            
        # Get the boundaries
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
                
        # Calculate padding based on the label size
        height_padding = int(label_np.shape[0] * padding_pct)  # Changed to use region height
        width_padding = int(label_np.shape[1] * padding_pct)   # Changed to use region width
        
        # Apply padding with bounds checking
        y_min = max(0, y_min - height_padding)
        y_max = min(label_np.shape[0], y_max + height_padding)
        x_min = max(0, x_min - width_padding)
        x_max = min(label_np.shape[1], x_max + width_padding)
                
        # Ensure the crop box is valid
        if x_min >= x_max or y_min >= y_max:
            return image, label
    
        # Perform the crop
        crop_box = (int(x_min), int(y_min), int(x_max), int(y_max))
        return image.crop(crop_box), label.crop(crop_box)
    
    else:
        # Original random_crop logic
        border_pct = 0.8
        image_width = image.size[0]
        image_height = image.size[1]

        border = int(min(image_width, image_height) * border_pct)
        
        crop_win_width = np.random.randint(image_width - border, image_width)
        crop_win_height = np.random.randint(image_height - border, image_height)
        random_region = (
            (image_width - crop_win_width) // 2, (image_height - crop_win_height) // 2, 
            (image_width + crop_win_width) // 2, (image_height + crop_win_height) // 2
        )
        
        return image.crop(random_region), label.crop(random_region)

def random_rotate(image, label, angle=15):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-angle, angle)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
    return image, label


def color_enhance(image):
    if random.random() > 0.4:
        bright_intensity = random.randint(5, 15) / 10.0
        image = ImageEnhance.Brightness(image).enhance(bright_intensity)
        contrast_intensity = random.randint(5, 15) / 10.0
        image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
        color_intensity = random.randint(0, 20) / 10.0
        image = ImageEnhance.Color(image).enhance(color_intensity)
        sharp_intensity = random.randint(0, 30) / 10.0
        image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image



# def random_gaussian(image, mean=0.1, sigma=0.35):
#     def gaussianNoisy(im, mean=mean, sigma=sigma):
#         for _i in range(len(im)):
#             im[_i] += random.gauss(mean, sigma)
#         return im

#     img = np.asarray(image)
#     width, height = img.shape
#     img = gaussianNoisy(img[:].flatten(), mean, sigma)
#     img = img.reshape([width, height])
#     return Image.fromarray(np.uint8(img))

def random_gaussian(img):
    img = np.array(img)
    if random.random() > 0.4:
        blurSize = random.choice([3,5,7,9,11])
        img = cv2.GaussianBlur(img, (blurSize,blurSize), 0)
    return Image.fromarray(img)


def random_pepper(img, N=0.0015):
    img = np.array(img)
    if random.random() > 0.5:
        noiseNum = int(N * img.shape[0] * img.shape[1])
        for i in range(noiseNum):
            randX = random.randint(0, img.shape[0] - 1)
            randY = random.randint(0, img.shape[1] - 1)
            img[randX, randY] = random.randint(0, 1) * 255
    return Image.fromarray(img)


def random_cutout(image, label, n_holes=8, length=100):
    """Apply random cutout augmentation to both image and mask."""
    if random.random() > 0.5:  # Apply 30% of the time
        image = np.array(image)
        label = np.array(label)
        h, w = image.shape[:2]
        
        for _ in range(n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            
            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)
            
            image[y1:y2, x1:x2] = 0
            label[y1:y2, x1:x2] = 0  # Also zero out the mask in the same region
            
        return Image.fromarray(image), Image.fromarray(label)
    return image, label
