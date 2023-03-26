
import os
import PIL
import h5py
import torch
import random
import numpy as np
import scipy.io as sio
from functools import reduce
from torchvision import transforms
from collections import OrderedDict
from torchvision.utils import save_image
from __data_root__ import __DATA_ROOT__
from torchvision.datasets.folder import default_loader


########################################################################################################################
# Abstract Dataset
########################################################################################################################

class DatasetStructured(object):
    """
    Super class for structured dataset.
    """
    def __init__(self, factors, supervised=False):
        """
        :param factors: Tuple of (factors_structure, factors_name):
            - factors_values: [tuple/list(f1v1, f1v2, ...) or int(n_fv), ...]
            - factors_name: [factor1_name, ...]
        :param supervised: Whether to generate labels from factors_structure or not.
        """
        ################################################################################################################
        # 1. Dataset information
        ################################################################################################################
        factors_values, factors_name = factors
        # (1) Factors structure
        factors_values = [tuple(range(v)) if isinstance(v, int) else v for v in factors_values]
        self._factors_values = factors_values
        # (2) Factors name
        assert factors_name is not None and len(factors_name) == len(set(factors_name))
        self._factors_name = factors_name
        ################################################################################################################
        # 2. Label & dataset.
        ################################################################################################################
        # Generate label.
        if supervised: self._label = self._generate_category_label()
        else: self._label = None
        # For dataset: to be reimplemented.

    def __repr__(self):
        r = "%s[" % self.__class__.__name__
        for n, n_fvs in zip(self._factors_name, self.n_factors_values): r += "%s@%d," % (n, n_fvs)
        r = r[:-1] + "]"
        # Return
        return r

    def __len__(self):
        return reduce((lambda _x, _y: _x*_y), self.n_factors_values)

    def subset(self, factors=None, supervised=False):
        """
        :param factors: { factor1_name: [factor1_value1, factor1_value2, ..., ], ... }
        :param supervised:
        :return:
        """
        raise NotImplementedError

    @property
    def factors(self):
        ret = OrderedDict()
        for n, fvs in zip(self._factors_name, self._factors_values): ret[n] = fvs
        return ret

    @property
    def n_factors_values(self):
        return [len(fvs) for fvs in self._factors_values]

    ####################################################################################################################
    # Load dataset & label
    ####################################################################################################################

    def _generate_category_label(self):
        """
            Given self._np_dataset in shape of (num_data, ...) with num_data=prod_j num_factor_j_values, label will
        be generated as (num_data, num_factors) with label[i, j] denotes the unit category label of j-th factor of
        i-th sample.
        :return: (num_data, num_factors)
        """
        # 1. Init result. (num_factor1_values, num_factor2_values, ..., num_factors)
        label = None
        # 2. Generate (num_values_factor1, ..., num_factor_j_values, ..., 1)
        for factor_index, num_factor_values in enumerate(self.n_factors_values):
            # (1) Generate a range of (num_factor_values, )
            factor_label = np.arange(start=0, stop=num_factor_values, step=1, dtype=np.uint8)
            # (2) Expand to aforementioned format
            for j in range(len(self.n_factors_values)):
                if j < factor_index: factor_label = np.expand_dims(factor_label, axis=0)
                if j > factor_index: factor_label = np.expand_dims(factor_label, axis=-1)
            factor_label = np.broadcast_to(factor_label, shape=self.n_factors_values)
            factor_label = np.expand_dims(factor_label, axis=-1)
            # (3) Save to result
            if label is None: label = factor_label
            else: label = np.concatenate((label, factor_label), axis=-1)
        # 3. Reshape of (num_data, num_factors)
        label = np.reshape(label, newshape=(-1, label.shape[-1]))
        # Return
        return label

    ####################################################################################################################
    # Get batch data
    ####################################################################################################################

    def __getitem__(self, item):
        raise NotImplementedError

    def get_observations_from_factors(self, factors):
        """
        :param factors: (batch, n_factors)
        :return Observations corresponding to factors.
        """
        raise NotImplementedError

    def random_visualize(self, save_path, to_torch='from_numpy'):
        # Randomly select a factor config.
        selected_factor = [random.choice(v) for v in self.factors.values()]
        # 1. Init images. (MAX_N_FVS*n_factors, C, H, W)
        images = []
        # 2. Traversal for each factor.
        factor_names = list(self.factors.keys())
        for factor_index in range(len(selected_factor)):
            # Get factors for current traversal.
            cur_factors = OrderedDict()
            for index, key in enumerate(factor_names):
                cur_factors[key] = self.factors[key] if index == factor_index else [selected_factor[index]]
            # Get Tensor. (n_cur_factor_values, C, H, W)
            data = self.subset(factors=cur_factors)[0:len(self)]
            if to_torch == 'from_numpy': data = torch.from_numpy(data)
            else: raise ValueError
            """ Saving """
            images.append(data)
        # 3. Get images.
        max_n_fvs = max([len(_x) for _x in images])
        images = torch.cat([_x if len(_x) == max_n_fvs else torch.cat([
            _x, torch.zeros(max_n_fvs-len(_x), *_x.size()[1:], dtype=_x.dtype, device=_x.device)]) for _x in images])
        """ Visualizing """
        save_dir = os.path.split(save_path)[0]
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        save_image(images, save_path, nrow=max_n_fvs)
        """ Printing factor names. """
        print("Factors: %s. " % factor_names)


class DatasetStructuredNumpy(DatasetStructured):
    """
    For numpy structured dataset.
    """
    def __init__(self, factors, dataset, supervised=False):
        """
        :param dataset: Will be np.array in the shape of (num_data, ...)
            - Str: Load from file.
            - np.array: Directly given.
        """
        # Set factors & labels.
        super(DatasetStructuredNumpy, self).__init__(factors, supervised)
        # Set dataset
        if isinstance(dataset, str): self._np_dataset = self._load_dataset_and_preproc(dataset_path=dataset)
        else: self._np_dataset = dataset
        assert len(self._np_dataset) == self.__len__()

    def subset(self, factors=None, supervised=False):
        """
        :param factors: { factor1_name: [factor1_value1, factor1_value2, ..., ], ... }
        :param supervised:
        :return:
        """
        if factors is None: return self
        ################################################################################################################
        # 1. Generate sub dataset.
        ################################################################################################################
        # (1) Reshape to structure
        np_dataset_structured = np.reshape(self._np_dataset, newshape=(tuple(self.n_factors_values) + self._np_dataset.shape[1:]))
        # (2) Get result: processing each factor.
        for factor_name, factor_values in factors.items():
            factor_index = self._factors_name.index(factor_name)
            # Get current result
            np_dataset_cur_factor = None
            for factor_v in factor_values:
                # Get subset
                np_dataset_subset = eval('np_dataset_structured[%s%s%s]' % (
                    ':,' * factor_index,
                    '%d:%d' % (factor_v, factor_v + 1),
                    ',:' * (len(np_dataset_structured.shape) - 1 - factor_index)))
                # Save to result
                if np_dataset_cur_factor is None: np_dataset_cur_factor = np_dataset_subset
                else: np_dataset_cur_factor = np.concatenate((np_dataset_cur_factor, np_dataset_subset), axis=factor_index)
            # Update
            np_dataset_structured = np_dataset_cur_factor
        # (3) Flatten
        np_dataset = np.reshape(np_dataset_structured, newshape=(-1, ) + self._np_dataset.shape[1:])
        ################################################################################################################
        # 2. Generator sub factors_info.
        ################################################################################################################
        # (1) Init results.
        factors_values, factors_name = [], []
        # (2) Process each factor.
        for f_name, f_values in zip(self._factors_name, self._factors_values):
            # Filter out only one factor value.
            if f_name in factors.keys() and len(factors[f_name]) == 1: continue
            """ Saving """
            factors_values.append(factors[f_name] if f_name in factors.keys() else f_values)
            factors_name.append(f_name)
        # Get subset.
        subset = self.__class__(factors=(tuple(factors_values), tuple(factors_name)), dataset=np_dataset, supervised=supervised)
        # Return
        return subset

    ####################################################################################################################
    # Load dataset & label
    ####################################################################################################################

    def _load_dataset_and_preproc(self, dataset_path):
        raise NotImplementedError

    ####################################################################################################################
    # Get batch data
    ####################################################################################################################

    def _data_preprocess(self, data):
        raise NotImplementedError

    def __getitem__(self, index):
        # 1. Get data & label
        data = self._np_dataset[index]
        label = self._label[index] if self._label is not None else None
        # 2. Preprocess
        data = self._data_preprocess(data)
        # Return
        return data if label is None else (data, label)

    def get_observations_from_factors(self, factors):
        # 1. Convert factors to indicies. (..., n_factors).
        # (1) Get factor bases. (n_factors, )
        n_factor_values = np.array(self.n_factors_values)
        factor_bases = np.prod(n_factor_values) / np.cumprod(n_factor_values)
        # (2) Convert.
        indicies = np.dot(factors, factor_bases).astype(np.int64)
        # 2. Get item.
        return self._data_preprocess(self._np_dataset[indicies])


########################################################################################################################
# Dataset Instances
########################################################################################################################

def _load_mesh_for_cars3d(filepath):
    """ Parses a single source file and rescales contained images. """
    mesh = np.einsum("abcde->deabc", sio.loadmat(filepath)["im"])
    flattened_mesh = mesh.reshape((-1,) + mesh.shape[2:])
    rescaled_mesh = np.zeros((flattened_mesh.shape[0], 64, 64, 3), dtype="uint8")
    for i in range(flattened_mesh.shape[0]):
        pic = PIL.Image.fromarray(flattened_mesh[i, :, :, :])
        pic.thumbnail((64, 64), PIL.Image.ANTIALIAS)
        rescaled_mesh[i, :, :, :] = np.array(pic)
    return rescaled_mesh


def _load_small_norb_chunks(dataset_dir):
    """ Loads several chunks of the small norb data set for final use. """
    list_of_images, list_of_features = _load_chunks(dataset_dir)
    features = np.concatenate(list_of_features, axis=0)
    features[:, 3] = features[:, 3] / 2  # azimuth values are 0, 2, 4, ..., 24
    return np.concatenate(list_of_images, axis=0)[:, np.newaxis], features


def _load_chunks(dataset_dir):
    """ Loads several chunks of the small norb data set into lists. """
    template = os.path.join(dataset_dir, "smallnorb-{}-{}.mat")
    # Loadding chunks.
    list_of_images = []
    list_of_features = []
    for chunk_name in ["5x46789x9x18x6x2x96x96-training", "5x01235x9x18x6x2x96x96-testing"]:
        norb = _read_binary_matrix(template.format(chunk_name, "dat"))
        list_of_images.append(_resize_images(norb[:, 0]))
        norb_class = _read_binary_matrix(template.format(chunk_name, "cat"))
        norb_info = _read_binary_matrix(template.format(chunk_name, "info"))
        list_of_features.append(np.column_stack((norb_class, norb_info)))
    return list_of_images, list_of_features


def _read_binary_matrix(filename):
    """ Reads and returns binary formatted matrix stored in filename. """
    with open(filename, "rb") as f:
        s = f.read()
        magic = int(np.frombuffer(s, "int32", 1))
        ndim = int(np.frombuffer(s, "int32", 1, 4))
        eff_dim = max(3, ndim)
        raw_dims = np.frombuffer(s, "int32", eff_dim, 8)
        dims = []
        for i in range(0, ndim):
            dims.append(raw_dims[i])

        dtype_map = {
            507333717: "int8",
            507333716: "int32",
            507333713: "float",
            507333715: "double"
        }
        data = np.frombuffer(s, dtype_map[magic], offset=8 + eff_dim * 4)
    data = data.reshape(tuple(dims))
    return data


def _resize_images(integer_images):
    resized_images = np.zeros((integer_images.shape[0], 64, 64), dtype="uint8")
    for i in range(integer_images.shape[0]):
        image = PIL.Image.fromarray(integer_images[i, :, :])
        image = image.resize((64, 64), PIL.Image.ANTIALIAS)
        resized_images[i, :, :] = image
    return resized_images


# ----------------------------------------------------------------------------------------------------------------------
# Structured numpy dataset (labelled)
# ----------------------------------------------------------------------------------------------------------------------

class Shapes(DatasetStructuredNumpy):
    """
    Shapes dataset.
    """
    def __init__(self, factors=((3, 6, 40, 32, 32), ('shape', 'scale', 'orientation', 'pos_x', 'pos_y')),
                 dataset=os.path.join(__DATA_ROOT__, 'dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'), supervised=False):
        super(Shapes, self).__init__(factors, dataset, supervised)

    def _load_dataset_and_preproc(self, dataset_path):
        dataset = np.load(dataset_path, encoding='latin1')['imgs'][:, np.newaxis]
        return dataset

    def _data_preprocess(self, data):
        return data.astype('float32')


class Shapes3D(DatasetStructuredNumpy):
    """
    3DShapes dataset.
    """
    def __init__(self, factors=((10, 10, 10, 8, 4, 15), ('floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation')),
                 dataset=os.path.join(__DATA_ROOT__, '3dshapes'), supervised=False):
        super(Shapes3D, self).__init__(factors, dataset, supervised)

    def _load_dataset_and_preproc(self, dataset_path):
        """
        :param dataset_path: Shapes3D dataset dir.
        :return:
        """
        npy_path = os.path.join(dataset_path, 'shapes3d_ndarray_fl10wa10ob10sc8sh4or15_3x64x64_byme.npy')
        if os.path.exists(npy_path): return np.load(npy_path)

        # --------------------------------------------------------------------------------------------------------------
        dataset = h5py.File(os.path.join(dataset_path, 'raw', '3dshapes.h5'), 'r')['images'][()].swapaxes(2, 3).swapaxes(1, 2)
        """ Saving """
        np.save(npy_path, dataset)
        # Return
        return dataset

    def _data_preprocess(self, data):
        return data.astype('float32') / 255.0


class Faces(DatasetStructuredNumpy):
    """
    Faces dataset.
    """
    def __init__(self, factors=((50, 21, 11, 11), ('face_id', 'azimuth', 'elevation', 'lighting')),
                 dataset=os.path.join(__DATA_ROOT__, '3dfaces/basel_face_renders.pth'), supervised=False):
        super(Faces, self).__init__(factors, dataset, supervised)

    def _load_dataset_and_preproc(self, dataset_path):
        dataset = torch.load(dataset_path).view(-1, 1, 64, 64).numpy()
        return dataset

    def _data_preprocess(self, data):
        return data.astype('float32') / 255.0


class Cars3D(DatasetStructuredNumpy):
    """
    Cars3D dataset.
    """
    def __init__(self, factors=((4, 24, 183), ('elevation', 'azimuth', 'object_type')),
                 dataset=os.path.join(__DATA_ROOT__, 'cars3d'), supervised=False):
        super(Cars3D, self).__init__(factors, dataset, supervised)

    def _load_dataset_and_preproc(self, dataset_path):
        """
        :param dataset_path: The Cars3D dataset dir.
        :return: Numpy.array. (183*24*4, 3, 64, 64)
        """
        npy_path = os.path.join(dataset_path, 'cars3d_ndarray_el4az24ob183_3x64x64_byme.npy')
        if os.path.exists(npy_path): return np.load(npy_path)

        # --------------------------------------------------------------------------------------------------------------
        dataset_path = os.path.join(dataset_path, 'raw')
        factor_bases = 4*24*183 / np.cumprod([4, 24, 183])
        # 1. Init results.
        dataset = np.zeros((4*24*183, 64, 64, 3), dtype="uint8")
        # 2. Load from each .mat file.
        for i, filename in enumerate([x for x in os.listdir(dataset_path) if ".mat" in x]):
            """ Loading """
            data_mesh = _load_mesh_for_cars3d(os.path.join(dataset_path, filename))
            # (1) Get factors. (factor1: elevation, factor2: azimuth, factor3: object_type)
            factor1, factor2 = np.array(list(range(4))), np.array(list(range(24)))
            all_factors = np.transpose([np.tile(factor1, len(factor2)), np.repeat(factor2, len(factor1)), np.tile(i, len(factor1)*len(factor2))])
            # (2) Get indices.
            indices = np.array(np.dot(all_factors, factor_bases), dtype=np.int64)
            """ Saving """
            dataset[indices] = data_mesh
        dataset = dataset.swapaxes(2, 3).swapaxes(1, 2)
        """ Saving """
        np.save(npy_path, dataset)
        # Return
        return dataset

    def _data_preprocess(self, data):
        return data.astype('float32') / 255.0


class SmallNorb(DatasetStructuredNumpy):
    """
    Small NORB dataset.
    """
    def __init__(self, factors=((5, 10, 9, 18, 6), ('category', 'instance', 'elevation', 'azimuth', 'lighting_condition')),
                 dataset=os.path.join(__DATA_ROOT__, 'small_norb'), supervised=False):
        super(SmallNorb, self).__init__(factors, dataset, supervised)

    def _load_dataset_and_preproc(self, dataset_path):
        """
        :param dataset_path: The SmallNORB dataset dir.
        :return: Numpy.array. (5*10*9*18*6, 1, 64, 64)
        """
        npy_path = os.path.join(dataset_path, 'smallnorb_ndarray_ca5in10el9az18li6_1x64x64_byme.npy')
        if os.path.exists(npy_path): return np.load(npy_path)

        # --------------------------------------------------------------------------------------------------------------
        dataset, factors = _load_small_norb_chunks(os.path.join(dataset_path, 'raw'))
        # Rearrange samples to be coincident with the structure (5, 10, 9, 18, 6)
        factor_bases = 5*10*9*18*6 / np.cumprod([5, 10, 9, 18, 6])
        indices = np.array(np.dot(factors, factor_bases), dtype=np.int64)
        # Get results.
        results = np.zeros_like(dataset)
        results[indices] = dataset
        """ Saving """
        np.save(npy_path, results)
        # Return
        return results

    def _data_preprocess(self, data):
        return data.astype('float32') / 255.0


# ----------------------------------------------------------------------------------------------------------------------
# Image dataset 64x64 (unlabelled)
# ----------------------------------------------------------------------------------------------------------------------

class ImageDataset(object):
    """
    Image dataset 64x64.
    """
    def __init__(self, root_dir):
        # Get paths.
        self._paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]
        # Get transforms
        self._transforms = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

    def __len__(self):
        return len(self._paths)

    def __getitem__(self, index):
        # 1. Load
        img = default_loader(self._paths[index])
        # 2. Transform
        img = self._transforms(img)
        # Return
        return img


class CelebA(ImageDataset):
    """
    CelebA dataset.
    """
    def __init__(self):
        super(CelebA, self).__init__(os.path.join(__DATA_ROOT__, "celeba/Img/img_align_celeba_png"))
    
        
class Chairs(ImageDataset):
    """
    3DChairs dataset.
    """
    def __init__(self):
        super(Chairs, self).__init__(os.path.join(__DATA_ROOT__, "3dchairs/images"))
