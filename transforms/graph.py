import torchio as tio
import matplotlib.pyplot as plt
from IPython import display
from tqdm.notebook import tqdm
from pathlib import Path


def plot_histogram(image_paths, compute_histograms):

    paths = image_paths

    if compute_histograms:
        fig, ax = plt.subplots(dpi=100)
        for path in tqdm(paths):
            tensor, _ = tio.ScalarImage(path).data
            if 'HH' in path.name: color = 'red'
            elif 'Guys' in path.name: color = 'green'
            elif 'IOP' in path.name: color = 'blue'
            plot_histogram(ax, tensor, color=color)
        ax.set_xlim(-100, 2000)
        ax.set_ylim(0, 0.004);
        ax.set_title('Original histograms of all samples')
        ax.set_xlabel('Intensity')
        ax.grid()
        graph = None

    # graph

    # landmarks = tio.HistogramStandardization.train(
    #     image_paths,
    #     output_path=histogram_landmarks_path,
    # )
    # tio.np.set_printoptions(suppress=True, precision=3)
    # print('\nTrained landmarks:', landmarks)

def main():
    dataset_dir_name = r'C:\Pycharmclass\NetworkProgramming\Torch\Data\ixi_tiny'
    dataset_dir = Path(dataset_dir_name)
    images_dir = dataset_dir / 'image'
    labels_dir = dataset_dir / 'label'
    image_paths = sorted(images_dir.glob('*.nii.gz'))

    plot_histogram(image_paths, compute_histograms=True)

if __name__ == "__main__":
    main()

