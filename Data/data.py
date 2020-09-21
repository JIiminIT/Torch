import enum
from pathlib import Path
import torchio as tio
import torch
from torch import multiprocessing
from torch.utils.data import dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
import nibabel as nib
import copy
from torchio import AFFINE, DATA
import numpy as np
import SimpleITK as sitk
import tempfile
from scipy import stats
from unet import UNet
import torch.nn.functional as F

from torchio.transforms import (
    RandomFlip,
    RandomAffine,
    RandomElasticDeformation,
    RandomNoise,
    RandomMotion,
    RandomBiasField,
    RescaleIntensity,
    Resample,
    ToCanonical,
    ZNormalization,
    CropOrPad,
    HistogramStandardization,
    OneOf,
    Compose,
)



def show_nifti(image_path_or_image, colormap='gray'):
    try:
        from niwidgets import NiftiWidget
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            widget = NiftiWidget(image_path_or_image)
            widget.nifti_plotter(colormap=colormap)

    except Exception:
        if isinstance(image_path_or_image, nib.AnalyzeImage):
            nii = image_path_or_image
        else:
            image_path = image_path_or_image
            nii = nib.load(str(image_path))
        k = int(nii.shape[-1] / 2)
        plt.imshow(nii.dataobj[..., k], cmap=colormap)
        plt.show()
def show_sample(sample, image_name, label_name=None):
    if label_name is not None:
        sample = copy.deepcopy(sample)
        affine = sample[label_name][AFFINE]
        label = sample[label_name][DATA][0].numpy().astype(np.uint8)
        label = np.expand_dims(label, axis=0)
        label_image = tio.utils.nib_to_sitk(label, affine)
        # label_image = np.squeeze(label_image)
        border = sitk.BinaryContour(label_image)
        border_array, _ = tio.utils.sitk_to_nib(border)
        border_array = np.squeeze(border_array)
        border_tensor = torch.from_numpy(border_array)
        image_tensor = sample[image_name][DATA][0]
        # border_tensor = np.squeeze(border_tensor)
        # print(image_tensor.max(), image_tensor.min())
        # print(border_tensor.shape)
        image_tensor[border_tensor > 0.5] = image_tensor.max()
        tensor = sample[image_name][DATA][0]
        affine = sample[image_name][AFFINE]
        nib_wrap = nib.Nifti1Image(np.transpose(tensor.numpy(), [2, 1, 0]), affine)
        nib.save(nib_wrap, 'tmp.nii.gz')
        show_nifti('tmp.nii.gz')


def save_image(sample, image_name, filename):
    tensor = sample[image_name][DATA][0]
    affine = sample[image_name][AFFINE]
    nib_wrap = nib.Nifti1Image(tensor.numpy(), affine)
    nib.save(nib_wrap, filename)


def plot_histogram(axis, tensor, num_positions=100, label=None, alpha=0.05, color=None):
    values = tensor.numpy().ravel()
    kernel = stats.gaussian_kde(values)
    positions = np.linspace(values.min(), values.max(), num=num_positions)
    histogram = kernel(positions)
    kwargs = dict(linewidth=1, color='black' if color is None else color, alpha=alpha)
    if label is not None:
        kwargs['label'] = label
    axis.plot(positions, histogram, **kwargs)

# Downloaded files
# dataset_url = 'https://www.dropbox.com/s/ogxjwjxdv5mieah/ixi_tiny.zip?dl=0'
# dataset_path = 'ixi_tiny.zip'

dataset_dir_name = r'C:\Pycharmclass\NetworkProgramming\Torch\Data\ixi_tiny'
dataset_dir = Path(dataset_dir_name)
histogram_landmarks_path = 'landmarks.npy'

# if not dataset_dir.is_dir():
#     !curl --silent --output {dataset_path} --location {dataset_url}
#     !unzip -qq {dataset_path}
#     !tree -d {dataset_dir_name}

### [`SubjectsDataset`](https://torchio.readthedocs.io/data/dataset.html)


# subjectsDataset
def SubjectsDataset():
    images_dir = dataset_dir / 'image'
    labels_dir = dataset_dir / 'label'
    image_paths = sorted(images_dir.glob('*.nii.gz'))
    label_paths = sorted(labels_dir.glob('*.nii.gz'))
    assert len(image_paths) == len(label_paths)

    subjects = []
    for (image_path, label_path) in zip(image_paths, label_paths):
        subject = tio.Subject(
            mri=tio.ScalarImage(image_path),
            brain=tio.LabelMap(label_path),
        )
        subjects.append(subject)
       # subjects = np.array(subjects)
    dataset = tio.SubjectsDataset(subjects)
    print('Dataset size:', len(dataset), 'subjects')  ## => Dataset size : 566 subjects
    return dataset, subjects




# def show_dataset(param, param1, label_name):
#     dataset = SubjectsDataset()
#     one_subject = dataset[0]
#     print(one_subject)
#     print(one_subject.mri)



def histogram(image_paths, compute_histograms):
    paths = image_paths

    if compute_histograms:
        fig, ax = plt.subplots(dpi=100)
        for path in tqdm(paths):
            tensor = tio.ScalarImage(path).data
            if 'HH' in path.name: color = 'red'
            elif 'Guys' in path.name: color = 'green'
            elif 'IOP' in path.name: color = 'blue'
            plot_histogram(ax, tensor, color=color)
        ax.set_xlim(-100, 2000)
        ax.set_ylim(0, 0.004);
        ax.set_title('Original histograms of all samples')
        ax.set_xlabel('Intensity')
        ax.grid()
        plt.show()
        ax.show()
        # graph = None


def trained(image_paths):
    landmarks = tio.HistogramStandardization.train(
        image_paths,
        output_path=histogram_landmarks_path,
    )
    np.set_printoptions(suppress=True, precision=3)
    print('\nTrained landmarks:', landmarks)


def landmarks_histogram(dataset, landmarks, compute_histograms):
    landmarks_dict = {'mri': landmarks}
    histogram_transform = tio.HistogramStandardization(landmarks_dict)

    if compute_histograms:
        fig, ax = plt.subplots(dpi=100)
        for i, sample in enumerate(tqdm(dataset)):
            standard = histogram_transform(sample)
            tensor = standard.mri.data
            path = str(sample.mri.path)
            if 'HH' in path: color = 'red'
            elif 'Guys' in path: color = 'green'
            elif 'IOP' in path: color = 'blue'
            plot_histogram(ax, tensor, color=color)
        ax.set_xlim(0, 150)
        ax.set_ylim(0, 0.02)
        ax.set_title('Intensity values of all samples after histogram standardization')
        ax.set_xlabel('Intensity')
        ax.grid()
        plt.show()
        ax.show()
        #graph = None


def normalization(histogram_transform,dataset):
    znorm_transform = tio.ZNormalization(masking_method=tio.ZNormalization.mean)

    sample = dataset[0]
    transform = tio.Compose([histogram_transform, znorm_transform])
    znormed = transform(sample)

    fig, ax = plt.subplots(dpi=100)
    plot_histogram(ax, znormed.mri.data, label='Z-normed', alpha=1)
    ax.set_title('Intensity values of one sample after z-normalization')
    ax.set_xlabel('Intensity')
    ax.grid()
    plt.show()
    ax.show()




def training_network(landmarks, dataset, subjects):
    training_transform = Compose([
        ToCanonical(),
        Resample(4),
        CropOrPad((48, 60, 48), padding_mode='reflect'),
        RandomMotion(),
        HistogramStandardization({'mri': landmarks}),
        RandomBiasField(),
        ZNormalization(masking_method=ZNormalization.mean),
        RandomNoise(),
        RandomFlip(axes=(0,)),
        OneOf({
            RandomAffine(): 0.8,
            RandomElasticDeformation(): 0.2,
        }),
    ])

    validation_transform = Compose([
        ToCanonical(),
        Resample(4),
        CropOrPad((48, 60, 48), padding_mode='reflect'),
        HistogramStandardization({'mri': landmarks}),
        ZNormalization(masking_method=ZNormalization.mean),
    ])

    training_split_ratio = 0.9
    num_subjects = len(dataset)
    num_training_subjects = int(training_split_ratio * num_subjects)

    training_subjects = subjects[:num_training_subjects]
    validation_subjects = subjects[num_training_subjects:]

    training_set = tio.SubjectsDataset(
        training_subjects, transform=training_transform)

    validation_set = tio.SubjectsDataset(
        validation_subjects, transform=validation_transform)

    print('Training set:', len(training_set), 'subjects')
    print('Validation set:', len(validation_set), 'subjects')
    return training_set, validation_set



device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
CHANNELS_DIMENSION = 1
SPATIAL_DIMENSIONS = 2, 3, 4

class Action(enum.Enum):
    TRAIN = 'Training'
    VALIDATE = 'Validation'

def prepare_batch(batch, device):
    inputs = batch['mri'][DATA].to(device)
    foreground = batch['brain'][DATA].to(device)
    background = 1 - foreground
    targets = torch.cat((background, foreground), dim=CHANNELS_DIMENSION)
    return inputs, targets

def get_dice_score(output, target, epsilon=1e-9):
    p0 = output
    g0 = target
    p1 = 1 - p0
    g1 = 1 - g0
    tp = (p0 * g0).sum(dim=SPATIAL_DIMENSIONS)
    fp = (p0 * g1).sum(dim=SPATIAL_DIMENSIONS)
    fn = (p1 * g0).sum(dim=SPATIAL_DIMENSIONS)
    num = 2 * tp
    denom = 2 * tp + fp + fn + epsilon
    dice_score = num / denom
    return dice_score

def get_dice_loss(output, target):
    return 1 - get_dice_score(output, target)

def forward(model, inputs):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        logits = model(inputs)
    return logits

def get_model_and_optimizer(device):
    model = UNet(
        in_channels=1,
        out_classes=2,
        dimensions=3,
        num_encoding_blocks=3,
        out_channels_first_layer=8,
        normalization='batch',
        upsampling_type='linear',
        padding=True,
        activation='PReLU',
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    return model, optimizer

def run_epoch(epoch_idx, action, loader, model, optimizer):
    is_training = action == Action.TRAIN
    epoch_losses = []
    model.train(is_training)
    for batch_idx, batch in enumerate(tqdm(loader)):
        inputs, targets = prepare_batch(batch, device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(is_training):
            logits = forward(model, inputs)
            probabilities = F.softmax(logits, dim=CHANNELS_DIMENSION)
            batch_losses = get_dice_loss(probabilities, targets)
            batch_loss = batch_losses.mean()
            if is_training:
                batch_loss.backward()
                optimizer.step()
            epoch_losses.append(batch_loss.item())
    epoch_losses = np.array(epoch_losses)
    print(f'{action.value} mean loss: {epoch_losses.mean():0.3f}')

def train(num_epochs, training_loader, validation_loader, model, optimizer, weights_stem):
    run_epoch(0, Action.VALIDATE, validation_loader, model, optimizer)
    for epoch_idx in range(1, num_epochs + 1):
        print('Starting epoch', epoch_idx)
        run_epoch(epoch_idx, Action.TRAIN, training_loader, model, optimizer)
        run_epoch(epoch_idx, Action.VALIDATE, validation_loader, model, optimizer)
        torch.save(model.state_dict(), f'{weights_stem}_epoch_{epoch_idx}.pth')


def learning_stuff1(training_set):
    training_instance = training_set[42]  # transform is applied in SubjectsDataset
    show_sample(training_instance, 'mri', label_name='brain')


def learning_stuff2(validation_set):
    validation_instance = validation_set[42]
    show_sample(validation_instance, 'mri', label_name='brain')

def whole_images(training_set, validation_set ):
    training_batch_size = 16
    validation_batch_size = 2 * training_batch_size

    training_loader = torch.utils.data.DataLoader(
        training_set,
        batch_size=training_batch_size,
        shuffle=True,
        num_workers=multiprocessing.cpu_count(),
    )

    validation_loader = torch.utils.data.DataLoader(
        validation_set,
        batch_size=validation_batch_size,
        num_workers=multiprocessing.cpu_count(),
    )

def visualize(training_loader):
    one_batch = next(iter(training_loader))



def main():
    # SubjectsDataset()
    # show_subject(tio.ToCanonical()(one_subject), 'mri', label_name='brain')
    dataset_dir_name = r'C:\Pycharmclass\NetworkProgramming\Torch\Data\ixi_tiny'
    dataset_dir = Path(dataset_dir_name)
    images_dir = dataset_dir / 'image'
    labels_dir = dataset_dir / 'label'
    image_paths = sorted(images_dir.glob('*.nii.gz'))

    #histogram(image_paths, compute_histograms=True)

    dataset, subject = SubjectsDataset()
    one_subject = dataset[0]
    # print(one_subject)
    # print(one_subject.mri)
    # show_sample(tio.ToCanonical()(one_subject), 'mri', label_name='brain')

    # show_sample(one_subject, 'mri', label_name='brain')

    save_image(one_subject, 'mri', 'jimin.nii.gz')

    #trained(image_paths)

    landmarks = np.load("C:\\Pycharmclass\\NetworkProgramming\\Torch\\Data\\landmarks.npy")
    landmarks_dict = {'mri': landmarks}
    histogram_transform = tio.HistogramStandardization(landmarks_dict)

    #landmarks_histogram(dataset, landmarks, compute_histograms=True)

    #normalization(histogram_transform, dataset)

    training_network(landmarks, dataset, subject)

    training_set, validation_set = training_network(landmarks, dataset, subject)

   # learning_stuff1(training_set)

   # learning_stuff2(validation_set)

    whole_images(training_set, validation_set)

    # training_loader = train()

    # visualize(training_loader)

if __name__ == "__main__":
    main()



