import os
import random
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
from torch.utils import data
from torchvision import transforms

from image_proc import preproc
from config import Config
from utils import path_to_image


Image.MAX_IMAGE_PIXELS = None       # remove DecompressionBombWarning
config = Config()
_class_labels_TR_sorted = (
    'Airplane, Ant, Antenna, Archery, Axe, BabyCarriage, Bag, BalanceBeam, Balcony, Balloon, Basket, BasketballHoop, Beatle, Bed, Bee, Bench, Bicycle, '
    'BicycleFrame, BicycleStand, Boat, Bonsai, BoomLift, Bridge, BunkBed, Butterfly, Button, Cable, CableLift, Cage, Camcorder, Cannon, Canoe, Car, '
    'CarParkDropArm, Carriage, Cart, Caterpillar, CeilingLamp, Centipede, Chair, Clip, Clock, Clothes, CoatHanger, Comb, ConcretePumpTruck, Crack, Crane, '
    'Cup, DentalChair, Desk, DeskChair, Diagram, DishRack, DoorHandle, Dragonfish, Dragonfly, Drum, Earphone, Easel, ElectricIron, Excavator, Eyeglasses, '
    'Fan, Fence, Fencing, FerrisWheel, FireExtinguisher, Fishing, Flag, FloorLamp, Forklift, GasStation, Gate, Gear, Goal, Golf, GymEquipment, Hammock, '
    'Handcart, Handcraft, Handrail, HangGlider, Harp, Harvester, Headset, Helicopter, Helmet, Hook, HorizontalBar, Hydrovalve, IroningTable, Jewelry, Key, '
    'KidsPlayground, Kitchenware, Kite, Knife, Ladder, LaundryRack, Lightning, Lobster, Locust, Machine, MachineGun, MagazineRack, Mantis, Medal, MemorialArchway, '
    'Microphone, Missile, MobileHolder, Monitor, Mosquito, Motorcycle, MovingTrolley, Mower, MusicPlayer, MusicStand, ObservationTower, Octopus, OilWell, '
    'OlympicLogo, OperatingTable, OutdoorFitnessEquipment, Parachute, Pavilion, Piano, Pipe, PlowHarrow, PoleVault, Punchbag, Rack, Racket, Rifle, Ring, Robot, '
    'RockClimbing, Rope, Sailboat, Satellite, Scaffold, Scale, Scissor, Scooter, Sculpture, Seadragon, Seahorse, Seal, SewingMachine, Ship, Shoe, ShoppingCart, '
    'ShoppingTrolley, Shower, Shrimp, Signboard, Skateboarding, Skeleton, Skiing, Spade, SpeedBoat, Spider, Spoon, Stair, Stand, Stationary, SteeringWheel, '
    'Stethoscope, Stool, Stove, StreetLamp, SweetStand, Swing, Sword, TV, Table, TableChair, TableLamp, TableTennis, Tank, Tapeline, Teapot, Telescope, Tent, '
    'TobaccoPipe, Toy, Tractor, TrafficLight, TrafficSign, Trampoline, TransmissionTower, Tree, Tricycle, TrimmerCover, Tripod, Trombone, Truck, Trumpet, Tuba, '
    'UAV, Umbrella, UnevenBars, UtilityPole, VacuumCleaner, Violin, Wakesurfing, Watch, WaterTower, WateringPot, Well, WellLid, Wheel, Wheelchair, WindTurbine, Windmill, WineGlass, WireWhisk, Yacht'
)
class_labels_TR_sorted = _class_labels_TR_sorted.split(', ')


class MyData(data.Dataset):
    def __init__(self, datasets, data_size, is_train=True):
        """
        Initialize the dataset handler.
        
        When rgb_labels=True and color_channel_map is provided, converts RGB label images
        to multi-channel tensor based on the color mapping.
        """
        # data_size is None when using dynamic_size or data_size is manually set to None (for inference in the original size).
        self.is_train = is_train
        self.data_size = data_size
        self.load_all = config.load_all
        self.device = config.device
        self.grayscale_input = config.grayscale_input
        self.rgb_labels = config.rgb_labels
        self.color_channel_map = config.color_channel_map
        self.num_output_channels = config.num_output_channels
        self.color_tolerance = config.color_tolerance
        valid_extensions = ['.png', '.jpg', '.PNG', '.JPG', '.JPEG']

        if self.is_train and config.auxiliary_classification:
            self.cls_name2id = {_name: _id for _id, _name in enumerate(class_labels_TR_sorted)}
        self.transform_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        # For standard grayscale labels
        self.transform_label = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        # For RGB labels that need conversion to multi-channel
        self.transform_rgb_label = transforms.Compose([
            transforms.ToTensor(),
        ])
        dataset_root = os.path.join(config.data_root_dir, config.task)
        # datasets can be a list of different datasets for training on combined sets.
        self.image_paths = []
        for dataset in datasets.split('+'):
            image_root = os.path.join(dataset_root, dataset, 'im')
            self.image_paths += [os.path.join(image_root, p) for p in os.listdir(image_root) if any(p.endswith(ext) for ext in valid_extensions)]
        self.label_paths = []
        for p in self.image_paths:
            for ext in valid_extensions:
                ## 'im' and 'gt' may need modifying
                p_gt = p.replace('/im/', '/gt/')[:-(len(p.split('.')[-1])+1)] + ext
                file_exists = False
                if os.path.exists(p_gt):
                    self.label_paths.append(p_gt)
                    file_exists = True
                    break
            if not file_exists:
                print('Not exists:', p_gt)

        if len(self.label_paths) != len(self.image_paths):
            set_image_paths = set([os.path.splitext(p.split(os.sep)[-1])[0] for p in self.image_paths])
            set_label_paths = set([os.path.splitext(p.split(os.sep)[-1])[0] for p in self.label_paths])
            print('Path diff:', set_image_paths - set_label_paths)
            raise ValueError(f"There are different numbers of images ({len(self.label_paths)}) and labels ({len(self.image_paths)})")

        if self.load_all:
            self.images_loaded, self.labels_loaded = [], []
            self.class_labels_loaded = []
            # for image_path, label_path in zip(self.image_paths, self.label_paths):
            for image_path, label_path in tqdm(zip(self.image_paths, self.label_paths), total=len(self.image_paths)):
                color_type = 'gray' if self.grayscale_input else 'rgb'
                _image = path_to_image(image_path, size=self.data_size, color_type=color_type)
                # If using grayscale but model expects RGB, duplicate the single channel to 3 channels
                if self.grayscale_input:
                    # Convert grayscale to 3-channel by duplicating the single channel
                    _image_array = np.array(_image)
                    _image_rgb = np.stack([_image_array, _image_array, _image_array], axis=2)
                    _image = Image.fromarray(_image_rgb)
                
                # Load label as RGB or grayscale based on config
                label_color_type = 'rgb' if self.rgb_labels else 'gray'
                _label = path_to_image(label_path, size=self.data_size, color_type=label_color_type)
                
                self.images_loaded.append(_image)
                self.labels_loaded.append(_label)
                self.class_labels_loaded.append(
                    self.cls_name2id[label_path.split('/')[-1].split('#')[3]] if self.is_train and config.auxiliary_classification else -1
                )

    def __getitem__(self, index):
        if self.load_all:
            image = self.images_loaded[index]
            label = self.labels_loaded[index]
            class_label = self.class_labels_loaded[index] if self.is_train and config.auxiliary_classification else -1
        else:
            color_type = 'gray' if self.grayscale_input else 'rgb'
            image = path_to_image(self.image_paths[index], size=self.data_size, color_type=color_type)
            # If using grayscale but model expects RGB, duplicate the single channel to 3 channels
            if self.grayscale_input:
                # Convert grayscale to 3-channel by duplicating the single channel
                image_array = np.array(image)
                image_rgb = np.stack([image_array, image_array, image_array], axis=2)
                image = Image.fromarray(image_rgb)
            
            # Load label as RGB or grayscale based on config
            label_color_type = 'rgb' if self.rgb_labels else 'gray'
            label = path_to_image(self.label_paths[index], size=self.data_size, color_type=label_color_type)
            
            class_label = self.cls_name2id[self.label_paths[index].split('/')[-1].split('#')[3]] if self.is_train and config.auxiliary_classification else -1

        # loading image and label (this must be done before label tensor conversion)
        if self.is_train:
            if config.background_color_synthesis:
                # For RGB labels, we should use the first channel only as the alpha mask
                # or convert the RGB mask to a single channel if not already processed
                alpha_mask = label
                if self.rgb_labels and isinstance(label, torch.Tensor):
                    # If label is already a multi-channel tensor, use the first channel as mask
                    alpha_mask = label[0].cpu().numpy() * 255
                    alpha_mask = Image.fromarray(alpha_mask.astype(np.uint8))
                
                image.putalpha(alpha_mask)
                array_image = np.array(image)
                array_foreground = array_image[:, :, :3].astype(np.float32)
                array_mask = (array_image[:, :, 3:] / 255).astype(np.float32)
                array_background = np.zeros_like(array_foreground)
                choice = random.random()
                if choice < 0.4:
                    # Black/Gray/White backgrounds
                    array_background[:, :, :] = random.randint(0, 255)
                elif choice < 0.8:
                    # Background color that similar to the foreground object. Hard negative samples.
                    foreground_pixel_number = np.sum(array_mask > 0)
                    if foreground_pixel_number > 0:  # Avoid division by zero
                        color_foreground_mean = np.mean(array_foreground * array_mask, axis=(0, 1)) * (np.prod(array_foreground.shape[:2]) / foreground_pixel_number)
                        color_up_or_down = random.choice((-1, 1))
                        # Up or down for 20% range from 255 or 0, respectively.
                        color_foreground_mean += (255 - color_foreground_mean if color_up_or_down == 1 else color_foreground_mean) * (random.random() * 0.2) * color_up_or_down
                        array_background[:, :, :] = color_foreground_mean
                    else:
                        # Fallback if mask is empty
                        array_background[:, :, :] = random.randint(0, 255)
                else:
                    # Any color
                    for idx_channel in range(3):
                        array_background[:, :, idx_channel] = random.randint(0, 255)
                array_foreground_background = array_foreground * array_mask + array_background * (1 - array_mask)
                image = Image.fromarray(array_foreground_background.astype(np.uint8))
                
            # If label is a tensor (already processed rgb), we skip preproc and process image only
            if isinstance(label, torch.Tensor):
                # Process image only and preserve the tensor label
                if config.preproc_methods:
                    for method in config.preproc_methods:
                        if method == 'flip' and random.random() > 0.5:
                            image = image.transpose(Image.FLIP_LEFT_RIGHT)
                            # Flip the label tensor
                            label = torch.flip(label, [2])  # Flip the width dimension
                        elif method == 'enhance':
                            image = color_enhance(image)
                        elif method == 'pepper':
                            image = random_pepper(image)
                        # Ignore rotate and crop as they would require more complex tensor operations
            else:
                # If label is still an image, process both normally
                image, label = preproc(image, label, preproc_methods=config.preproc_methods)
        # else:
        #     if _label.shape[0] > 2048 or _label.shape[1] > 2048:
        #         _image = cv2.resize(_image, (2048, 2048), interpolation=cv2.INTER_LINEAR)
        #         _label = cv2.resize(_label, (2048, 2048), interpolation=cv2.INTER_LINEAR)

        # At present, we use fixed sizes in inference, instead of consistent dynamic size with training.
        if self.is_train:
            if config.dynamic_size is None:
                image = self.transform_image(image)
                
                # For RGB labels, use rgb_to_multi_channel conversion after ToTensor
                if self.rgb_labels and self.color_channel_map:
                    # First convert to tensor
                    label_tensor = self.transform_rgb_label(label)
                    # Then convert RGB tensor to multi-channel with configured tolerance
                    label = self.rgb_to_multi_channel(label_tensor, self.color_tolerance)
                else:
                    # Standard grayscale label processing
                    label = self.transform_label(label)
        else:
            size_div_32 = (int(image.size[0] // 32 * 32), int(image.size[1] // 32 * 32))
            if image.size != size_div_32:
                image = image.resize(size_div_32)
                label = label.resize(size_div_32)
                
            image = self.transform_image(image)
            
            # For RGB labels, use rgb_to_multi_channel conversion after ToTensor
            if self.rgb_labels and self.color_channel_map:
                # First convert to tensor
                label_tensor = self.transform_rgb_label(label)
                # Then convert RGB tensor to multi-channel with configured tolerance
                label = self.rgb_to_multi_channel(label_tensor, self.color_tolerance)
            else:
                # Standard grayscale label processing
                label = self.transform_label(label)

        if self.is_train:
            return image, label, class_label
        else:
            return image, label, self.label_paths[index]

    def __len__(self):
        return len(self.image_paths)
    
    def rgb_to_multi_channel(self, rgb_label, color_tolerance=10):
        """
        Convert an RGB label tensor to a multi-channel tensor based on color mapping.
        
        Args:
            rgb_label: RGB tensor with shape [3, H, W] and values in range [0, 1]
            color_tolerance: Integer tolerance value for color matching. 
                             If >0, colors within this distance will match.
            
        Returns:
            Multi-channel tensor with shape [num_output_channels, H, W] where each channel
            contains a binary mask for pixels matching the corresponding color.
        """
        if not self.rgb_labels or not self.color_channel_map:
            # If RGB labels not enabled or no mapping provided, return as is
            return rgb_label
        
        # Get image dimensions
        _, h, w = rgb_label.shape
        
        # Create output tensor with num_output_channels
        multi_channel = torch.zeros((self.num_output_channels, h, w), dtype=torch.float32)
        
        # Convert RGB tensor to a scaled format for easier comparison (0-255)
        rgb_values = (rgb_label * 255).round().long()
        
        # Process each color in the mapping
        for (r, g, b), channel_idx in self.color_channel_map.items():
            # Check if channel index is valid
            if channel_idx >= self.num_output_channels:
                continue
            
            if color_tolerance > 0:
                # Approximate color matching with tolerance
                r_dist = torch.abs(rgb_values[0] - r)
                g_dist = torch.abs(rgb_values[1] - g)
                b_dist = torch.abs(rgb_values[2] - b)
                
                # Color matches if each channel is within tolerance
                r_match = (r_dist <= color_tolerance)
                g_match = (g_dist <= color_tolerance)
                b_match = (b_dist <= color_tolerance)
                
                # Or use a combined distance metric (Euclidean distance)
                # total_dist = torch.sqrt(r_dist.float().pow(2) + g_dist.float().pow(2) + b_dist.float().pow(2))
                # color_mask = (total_dist <= color_tolerance)
            else:
                # Exact color matching
                r_match = (rgb_values[0] == r)
                g_match = (rgb_values[1] == g)
                b_match = (rgb_values[2] == b)
            
            color_mask = r_match & g_match & b_match
            
            # Set the corresponding channel to 1 where the color matches
            multi_channel[channel_idx][color_mask] = 1.0
            
        return multi_channel


def custom_collate_fn(batch):
    if config.dynamic_size:
        dynamic_size = tuple(sorted(config.dynamic_size))
        dynamic_size_batch = (random.randint(dynamic_size[0][0], dynamic_size[0][1]) // 32 * 32, random.randint(dynamic_size[1][0], dynamic_size[1][1]) // 32 * 32) # select a value randomly in the range of [dynamic_size[0/1][0], dynamic_size[0/1][1]].
        data_size = dynamic_size_batch
    else:
        data_size = config.size
    new_batch = []
    use_grayscale = config.grayscale_input
    transform_image = transforms.Compose([
        transforms.Resize(data_size[::-1]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_label = transforms.Compose([
        transforms.Resize(data_size[::-1]),
        transforms.ToTensor(),
    ])
    
    # Create a dataset instance to use its rgb_to_multi_channel method
    rgb_labels = config.rgb_labels
    color_channel_map = config.color_channel_map
    num_output_channels = config.num_output_channels
    
    for image, label, class_label in batch:
        # Check if the label is already a tensor (processed) or not
        if isinstance(label, torch.Tensor):
            # Label was already processed during __getitem__
            # Just resize it if needed
            if label.shape[-2:] != tuple(data_size[::-1]):
                # For multi-channel labels, handle resizing for each channel
                resized_label = torch.nn.functional.interpolate(
                    label.unsqueeze(0), 
                    size=data_size[::-1], 
                    mode='nearest'
                ).squeeze(0)
                new_batch.append((transform_image(image), resized_label, class_label))
            else:
                new_batch.append((transform_image(image), label, class_label))
        else:
            # Label is still an image (PIL), process it
            new_batch.append((transform_image(image), transform_label(label), class_label))
            
    return data._utils.collate.default_collate(new_batch)
