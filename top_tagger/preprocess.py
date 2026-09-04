import os,uproot,awkward
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset


class JetDataSet(Dataset):

    """
    PyTorch Dataset wrapper for preprocessed jet data.

    Each jet is represented by particle-level features, particle
    four-momenta, a particle-validity mask, and a binary jet label.
    A slice of the input tensors can be selected to construct a
    training, validation, or test subset.

    Parameters
    ----------
    X_tensor : torch.Tensor
        Standardized particle-level features with shape
        ``(num_jets, max_particles, 7)``.

    P4_tensor : torch.Tensor
        Particle four-momenta with shape
        ``(num_jets, max_particles, 4)``.
        The four components are ``(px, py, pz, energy)``.

    mask_tensor : torch.Tensor
        Particle-validity mask with shape
        ``(num_jets, max_particles)``.
        ``True`` indicates a real particle and ``False`` indicates
        a padded particle.

    label_tensor : torch.Tensor
        Binary jet labels with shape ``(num_jets,)``.
        ``0`` corresponds to QCD jets and ``1`` corresponds to top jets.

    slice_lim : slice, optional
        Slice used to select a subset of the complete dataset.
        This can be used to construct training, validation, and test
        datasets. By default, the complete dataset is selected.

    Returns
    -------
    dict
        Each call to ``__getitem__`` returns a dictionary containing:

        ``"features"``
            Particle features with shape ``(max_particles, 7)``.

        ``"4-momentum"``
            Particle four-momenta with shape ``(max_particles, 4)``.

        ``"Mask"``
            Particle-validity mask with shape ``(max_particles,)``.

        ``"Jet Label"``
            Scalar binary jet label.
    """

    def __init__(self,
                 X_tensor: torch.Tensor,
                 P4_tensor:torch.Tensor,
                 mask_tensor:torch.Tensor,
                 label_tensor:torch.Tensor,
                 slice_lim : slice = slice(None)):

        self.x = X_tensor[slice_lim]
        self.p4 = P4_tensor[slice_lim]
        self.mask = mask_tensor[slice_lim]
        self.label = label_tensor[slice_lim]

    def __getitem__(self, idx):
        return {"features": self.x[idx],
                "4-momentum" : self.p4[idx],
                "Mask"  : self.mask[idx],
                "Jet Label" : self.label[idx]} 
        
    def __len__(self):
        return len(self.label)

    def __repr__(self):
        return (f"X : {self.x.shape}, P4 : {self.p4.shape}, Mask : {self.mask.shape}, Label : {self.label.shape}")


def prepare_jet_data(path,
                     max_particles,
                     max_sample_per_class,
                     seed = 0,
                     eps = 1e-8): 
    """
    Load and preprocess a balanced top/QCD jet dataset from a ROOT file.

    The function selects a fixed number of QCD and top jets, constructs
    seven particle-level features, pads each jet to a fixed number of
    particles, constructs particle masks and four-momenta, standardizes
    the features, and randomly shuffles the resulting dataset.

    Parameters
    ----------
    path : str
        Directory containing the ``jet_data.root`` ROOT file.

    max_particles : int
        Maximum number of particles retained per jet. Jets with fewer
        particles are zero-padded, while jets with more particles are
        truncated.

    max_sample_per_class : int
        Number of jets selected from each class. The resulting dataset
        contains ``2 * max_sample_per_class`` jets.

    seed : int, optional
        Random seed used for shuffling the dataset. Default is 0.

    eps : float, optional
        Small positive constant added inside logarithms and to the
        feature standard deviations to avoid numerical issues such as
        ``log(0)`` and division by zero. Default is ``1e-8``.

    Returns
    -------
    X_tensor : torch.Tensor
        Standardized particle-level features with shape
        ``(num_jets, max_particles, 7)``.

        The seven features are:
        ``deta``, ``dphi``, ``log_pt``, ``log_energy``,
        ``log_pt_rel``, ``log_energy_rel``, and ``dR``.

    P4_tensor : torch.Tensor
        Particle four-momenta with shape
        ``(num_jets, max_particles, 4)``.
        The four components are ``(px, py, pz, energy)``.

    mask_tensor : torch.Tensor
        Particle-validity mask with shape
        ``(num_jets, max_particles)``.
        ``True`` indicates a real particle and ``False`` indicates
        a padded particle.

    label_tensor : torch.Tensor
        Binary jet labels with shape ``(num_jets,)``.
        ``0`` corresponds to QCD jets and ``1`` corresponds to top jets.
    """
    
    file = uproot.open(os.path.join(path,"jet_data.root"))
    tree = file["tree"]
    # branches = tree.keys()
    data = tree.arrays(library="ak")
    required_branch_name = [
    "part_px",
    "part_py",
    "part_pz",
    "part_energy",
    "part_deta",
    "part_dphi"
    ]
    required_branch_data = data[required_branch_name]

    is_top = awkward.to_numpy(data["label_Tbl"]|data["label_Tbqq"]).astype(bool)
    is_qcd = awkward.to_numpy(data["label_QCD"]).astype(bool)
    jet_index = np.concatenate((np.where(is_qcd)[0][:max_sample_per_class],
                                np.where(is_top)[0][:max_sample_per_class]))

    jet_labels = np.concatenate((np.zeros(max_sample_per_class),np.ones(max_sample_per_class)))
    top_qcd_branch = required_branch_data[jet_index]

    px,py,pz,energy,deta,dphi = [top_qcd_branch[feat] for feat in required_branch_name]

    pt = np.sqrt(px**2+py**2)
    dR = np.sqrt(deta**2+dphi**2)
    pt_jet = np.sqrt((np.sum(px,axis = 1))**2 + (np.sum(py,axis = 1))**2)
    energy_jet = np.sum(energy,axis = 1)
    log_energy = np.log(energy + eps)
    log_pt = np.log(pt + eps)
    log_pt_rel = np.log(pt/pt_jet + eps)
    log_energy_rel = np.log(energy/energy_jet + eps)

    feature_list = [deta,dphi,log_pt,log_energy,log_pt_rel,log_energy_rel,dR]
    FEATURE_NAMES = ["deta","dphi","log_pt","log_energy","log_pt_rel","log_energy_rel","dR"]

    def pad_feat(feat_name):
        return awkward.to_numpy(awkward.fill_none(awkward.pad_none(feat_name,max_particles,axis = 1,clip = True),0.0,axis = 1))

    features = np.stack([pad_feat(feat) for feat in feature_list],axis = -1).astype(np.float32)
    P4 = np.stack([pad_feat(feat) for feat in [px,py,pz,energy]],axis = -1).astype(np.float32)
    nparts = awkward.to_numpy(awkward.num(pt,axis=1))
    mask = (np.arange(max_particles)[None,:]< nparts[:,None])

    features_mean,features_std = features[mask].mean(axis=0),features[mask].std(axis = 0)

    X = (features - features_mean)/(features_std + eps)

    X_tensor , P4_tensor , mask_tensor , label_tensor = map(torch.from_numpy,[X,P4,mask,jet_labels])

    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(label_tensor),generator=gen)
    X_tensor , P4_tensor , mask_tensor , label_tensor = X_tensor[perm] , P4_tensor[perm] , mask_tensor[perm] , label_tensor[perm]

    return X_tensor , P4_tensor , mask_tensor , label_tensor


def create_loader(jet_dataset: JetDataSet,train_frac,test_frac,batch_size = 64):

    """
    Split a JetDataSet into training, testing, and validation subsets
    and wrap each subset in a PyTorch DataLoader.

    Parameters
    ----------
    jet_dataset : JetDataSet
        Dataset containing the complete set of jets. Each sample contains:
            - "features"   : standardized particle features, shape (num_jets, N, 7)
            - "4-momentum" : particle four-momenta, shape (num_jets, N, 4)
            - "Mask"       : particle-validity mask, shape (num_jets, N)
            - "Jet Label"  : binary jet classification label. (num_jets,)

    train_frac : float
        Fraction of the total dataset assigned to the training set.

    test_frac : float
        Fraction of the total dataset assigned to the test set.
        The remaining fraction is assigned to validation.

    batch_size : int, optional
        Number of jets contained in each batch returned by the DataLoaders.
        Default is 64.

    Returns
    -------
    dict[str, DataLoader]
        Dictionary containing three DataLoaders:

            "train" : DataLoader
                Training data, shuffled at the beginning of each epoch.

            "test" : DataLoader
                Test data, not shuffled.

            "val" : DataLoader
                Validation data, not shuffled.

        Each batch is a dictionary with the same keys as JetDataSet.__getitem__:
            batch["features"]     -> (B, N, 7)
            batch["4-momentum"]   -> (B, N, 4)
            batch["Mask"]         -> (B, N)
            batch["Jet Label"]    -> (B,)

        where B is the batch size and N is the maximum number of
        particles per jet.
    """

    num_jets = len(jet_dataset)
    n_train = int(train_frac*num_jets)
    n_test = int(test_frac*num_jets)

    split = {
        "train" : slice(0,n_train),
        "test" : slice(n_train,n_test+n_train),
        "val" : slice(n_train+n_test,num_jets)
    }

    loaders = { key : DataLoader(JetDataSet(
        jet_dataset.x,
        jet_dataset.p4,
        jet_dataset.mask,
        jet_dataset.label,
        value
    ),batch_size=batch_size,shuffle = key=="train") for key,value in split.items()}

    print({key: len(JetDataSet(
            jet_dataset.x,
            jet_dataset.p4,
            jet_dataset.mask,
            jet_dataset.label,
            s)) for key, s in split.items()})
    
    return loaders


def main():
    X,P4,mask,label = prepare_jet_data(path = "/home/tuhin/Python codes and all/",
                                   max_particles = 128,
                                   max_sample_per_class = 8000,
                                   seed = 0)

    jet_dataset = JetDataSet(X_tensor= X,
                            P4_tensor = P4,
                            mask_tensor = mask,
                            label_tensor = label)

    print(jet_dataset[0]["features"].shape)

    loader = create_loader(jet_dataset=jet_dataset,
                        train_frac = 0.7,
                        test_frac = 0.15)

if __name__ == "__main__":
    main()

