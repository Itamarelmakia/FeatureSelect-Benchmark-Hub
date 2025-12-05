import copy
import numpy as np
import torch_geometric.nn as gnn
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
import random as rn
import os

# Assuming GRACES and all necessary imports and classes are defined elsewhere in the code

def GRACES_FS(p_train_feature, p_train_label,  k, row, random_state,device):
    """
    Perform feature selection based on the GRACES algorithm.
    
    Args:
        data_name (str): Name of the dataset.
        p_train_feature (numpy array): Training data with features.
        p_train_label (numpy array): Labels corresponding to the training data.
        p_test_feature (numpy array): Test data with features.
        p_test_label (numpy array): Labels corresponding to the test data.
        key_feature_number (int): Maximum number of top features to select.
        row (dict): Dictionary containing hyperparameters for the GRACES algorithm.
        random_state (int): Seed used by the random number generator for reproducibility.
    

    Returns:
        list: Indices of the top 'k' important features.
    """
        
    # Set seed for reproducibility
    seed = random_state
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    rn.seed(seed)


    # Initialize and configure the GRACES feature selector
    graces = GRACES(n_features=k,hidden_size=[64, 32], q=row['q'], n_dropouts=row['n_dropouts'], dropout_prob=row['dropout_prob'], batch_size=row['batch_size'],   learning_rate=row['learning_rate'], epochs=row['epochs'], alpha=row['alpha'], sigma=row['sigma'], f_correct=row['f_correct'])
    # Perform feature selection
    selected_features = graces.select(p_train_feature, p_train_label)

    # Calculate feature importance using RandomForestClassifier
    #sorted_indices,sorted_importance,accumulated_importance = graces.calculate_feature_importance_rf(p_train_feature, p_train_label, top_k_indices)

    return selected_features


# Example of how to call GRACES_FS
# results = GRACES_FS(data_name, train_features, train_labels, test_features, test_labels, 5, hyperparams_dict, 42)
# print(results)


class GraphConvNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, alpha):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.input = nn.Linear(self.input_size, self.hidden_size[0], bias=False)
        self.alpha = alpha
        self.hiddens = nn.ModuleList(
            [gnn.SAGEConv(self.hidden_size[h], self.hidden_size[h + 1]) for h in range(len(self.hidden_size) - 1)])
        self.output = nn.Linear(hidden_size[-1], output_size)

    def forward(self, x):
        edge_index = self.create_edge_index(x)
        x = self.input(x)
        x = self.relu(x)
        for hidden in self.hiddens:
            x = hidden(x, edge_index)
            x = self.relu(x)
        x = self.output(x)
        x = self.softmax(x)
        return x

    def create_edge_index(self, x):
        n = x.size(0)
        adj_matrix = torch.zeros((n, n), dtype=torch.bool, device=x.device)

        for i in range(n):
            # Compute cosine similarity between the ith vector and all vectors
            cos_sim = torch.abs(F.cosine_similarity(x[i].view(1, -1), x, dim=1))

            # Determine threshold based on quantile
            eps = torch.quantile(cos_sim, self.alpha, interpolation='nearest')
            adj_matrix[i] = cos_sim >= eps

        # Get the indices of the non-zero elements
        row, col = torch.where(adj_matrix)
        edge_index = torch.stack([row, col], dim=0)

        return edge_index

    @property
    def device(self):
        return next(self.parameters()).device



class GRACES:
    def __init__(self, n_features, hidden_size, q, n_dropouts, dropout_prob, batch_size,
                 learning_rate, epochs, alpha, sigma, f_correct): #Hyperparameters
        
        self.hidden_size = hidden_size
        self.n_features = n_features
        self.q = q
        self.n_dropouts = n_dropouts
        self.dropout_prob = dropout_prob
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.alpha = alpha
        self.sigma = sigma
        self.f_correct = f_correct
        self.S = None
        self.new = None
        self.model = None
        self.last_model = None
        self.loss_fn = None
        self.f_scores = None

    @staticmethod
    def bias(x):
        if not all(x[:, 0] == 1):
            x = torch.cat((torch.ones(x.shape[0], 1), x.float()), dim=1)
        return x

    def f_test(self, x, y):
        slc = SelectKBest(f_classif, k=x.shape[1])
        slc.fit(x, y)
        return getattr(slc, 'scores_')

    def xavier_initialization(self):
        if self.last_model is not None:
            weight = torch.zeros(self.hidden_size[0], len(self.S))
            nn.init.xavier_normal_(weight, gain=nn.init.calculate_gain('relu'))
            old_s = self.S.copy()
            if self.new in old_s:
                old_s.remove(self.new)
            for i in self.S:
                if i != self.new:
                    weight[:, self.S.index(i)] = self.last_model.input.weight.data[:, old_s.index(i)]
            self.model.input.weight.data = weight
            for h in range(len(self.hidden_size) - 1):
                self.model.hiddens[h].lin_l.weight.data = self.last_model.hiddens[h].lin_l.weight.data
                self.model.hiddens[h].lin_r.weight.data = self.last_model.hiddens[h].lin_r.weight.data
            self.model.output.weight.data = self.last_model.output.weight.data
    """
    def train(self, x, y):
        input_size = len(self.S)
        output_size = len(torch.unique(y))
        self.model = GraphConvNet(input_size, output_size, self.hidden_size, self.alpha)
        self.xavier_initialization()
        x = x[:, self.S]
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        train_set = []
        for i in range(x.shape[0]):
            train_set.append((x[i, :], y[i]))  # Ensure this is a tuple
        print(f"Train set sample: {train_set[0]}")  # Print the first sample to check format
        print(f"Batch size being used: {self.batch_size}")
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        for e in range(self.epochs):
            for data, label in train_loader:
                input_0 = data.view(data.shape[0], -1)
                optimizer.zero_grad()
                output = self.model(input_0.float())
                loss = self.loss_fn(output, label.long())
                loss.backward()
                optimizer.step()
        self.last_model = copy.deepcopy(self.model)
    """
    def train(self, x, y):
        input_size = len(self.S)
        output_size = len(torch.unique(y))
        self.model = GraphConvNet(input_size, output_size, self.hidden_size, self.alpha)
        self.xavier_initialization()
        x = x[:, self.S]
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate) 
        train_set = []
        for i in range(x.shape[0]):
            train_set.append([x[i, :], y[i]])
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size,
                                                   shuffle=True) 
        for e in range(self.epochs):
            for data, label in train_loader:
                input_0 = data.view(data.shape[0], -1)
                optimizer.zero_grad()
                output = self.model(input_0.float())
                loss = self.loss_fn(output, label.long())
                loss.backward()
                optimizer.step()
        self.last_model = copy.deepcopy(self.model)


    def dropout(self):
        model_dp = copy.deepcopy(self.model)
        for h in range(len(self.hidden_size) - 1):
            h_size = self.hidden_size[h]
            dropout_index = np.random.choice(range(h_size), int(h_size * self.dropout_prob), replace=False)
            model_dp.hiddens[h].lin_l.weight.data[:, dropout_index] = torch.zeros(
                model_dp.hiddens[h].lin_l.weight[:, dropout_index].shape)
            model_dp.hiddens[h].lin_r.weight.data[:, dropout_index] = torch.zeros(
                model_dp.hiddens[h].lin_r.weight[:, dropout_index].shape)
        dropout_index = np.random.choice(range(self.hidden_size[-1]), int(self.hidden_size[-1] * self.dropout_prob),
                                         replace=False)
        model_dp.output.weight.data[:, dropout_index] = torch.zeros(model_dp.output.weight[:, dropout_index].shape)
        return model_dp

    def gradient(self, x, y, model):
        model_gr = GraphConvNet(x.shape[1], len(torch.unique(y)), self.hidden_size, self.alpha)
        temp = torch.zeros(model_gr.input.weight.shape)
        temp[:, self.S] = model.input.weight
        model_gr.input.weight.data = temp
        for h in range(len(self.hidden_size) - 1):
            model_gr.hiddens[h].lin_l.weight.data = model.hiddens[h].lin_l.weight + self.sigma * torch.randn(
                model.hiddens[h].lin_l.weight.shape)
            model_gr.hiddens[h].lin_r.weight.data = model.hiddens[h].lin_r.weight + self.sigma * torch.randn(
                model.hiddens[h].lin_r.weight.shape)
        model_gr.output.weight.data = model.output.weight
        output_gr = model_gr(x.float())
        loss_gr = self.loss_fn(output_gr, y.long() )
        loss_gr.backward()
        input_gradient = model_gr.input.weight.grad
        return input_gradient

    def average(self, x, y, n_average):
        grad_cache = None
        for num in range(n_average):
            model = self.dropout()
            input_grad = self.gradient(x, y, model)
            if grad_cache is None:
                grad_cache = input_grad
            else:
                grad_cache += input_grad
        return grad_cache / n_average

    def find(self, input_gradient):
        gradient_norm = input_gradient.norm(p=self.q, dim=0)
        gradient_norm = gradient_norm / gradient_norm.norm(p=2)
        gradient_norm[1:] = (1 - self.f_correct) * gradient_norm[1:] + self.f_correct * self.f_scores
        gradient_norm[self.S] = 0
        max_index = torch.argmax(gradient_norm)
        return max_index.item()



    def select(self, x, y):
            x = np.array(x, dtype=np.float32)
            x = torch.tensor(x)
            y = torch.tensor(y)
            self.f_scores = torch.tensor(self.f_test(x, y))
            self.f_scores[torch.isnan(self.f_scores)] = 0
            self.f_scores = self.f_scores / self.f_scores.norm(p=2)
            x = self.bias(x)
            self.S = [0]
            self.loss_fn = nn.CrossEntropyLoss()
            while len(self.S) < self.n_features + 1:
                self.train(x, y)
                input_gradient = self.average(x, y, self.n_dropouts)
                self.new = self.find(input_gradient)
                self.S.append(self.new)
            selection = self.S
            selection.remove(0)
            selection = [s - 1 for s in selection]
            return selection
    
    # We add this function because the authors did not provide feature importance.
    # If we use the gradient method, we will not obtain the feature importance with respect to the sorted indices.
    def calculate_feature_importance_rf(self, x, y, selected_indices):
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np

        # Convert to numpy arrays if necessary
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        if isinstance(y, torch.Tensor):
            y = y.numpy()

        # Train a RandomForestClassifier on all features
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(x, y)

        # Extract feature importances for all features
        importances_all = clf.feature_importances_

        # Normalize the importances by the sum of all feature importances
        normalized_importances_all = importances_all / np.sum(importances_all)

        # Extract and sort the normalized importances for the selected features
        selected_importances = normalized_importances_all[selected_indices]
        sorted_indices_importance = sorted(zip(selected_indices, selected_importances), key=lambda x: x[1], reverse=True)
        sorted_indices, sorted_importance = zip(*sorted_indices_importance)

        # Round the importances to 3 decimal places
        sorted_importance_rounded = np.round(sorted_importance, 3)

        # Calculate accumulated importance
        accumulated_importance = np.round(np.sum(sorted_importance_rounded), 3)
        if accumulated_importance > 1:
            accumulated_importance = 1  # Adjust if sum exceeds 1 due to rounding

        return list(sorted_indices), list(sorted_importance_rounded), accumulated_importance