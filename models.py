""" Componets of the model
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
from SelfAttention import ScaledDotProductAttention
from torch_geometric.nn import GATConv
import dgl.nn.pytorch as dglnn

def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)


class HGCN(nn.Module):
    def __init__(self, in_feats, out_feats,hid_feats, dropout, rel_names):
        super().__init__()
        self.line1=LinearLayer(in_feats[0],out_feats)
        self.line2 = LinearLayer(in_feats[1], out_feats)
        self.HGCN_convolution=HGCN_convolution(out_feats, hid_feats, dropout, rel_names)
    def forward(self, graph):
        h=graph.ndata['feat']
        h['sam']=self.line1(h['sam'])
        h['gen']=self.line2(h['gen'])
        h=self.HGCN_convolution(graph,h)
        return h




class HGCN_convolution(nn.Module):
    def __init__(self, in_feats, hid_feats, dropout, rel_names):
        super().__init__()
        self.dropout=dropout
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats[0])
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats[0], hid_feats[1])
            for rel in rel_names}, aggregate='sum')
        self.conv3 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats[1], hid_feats[2])
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = {k: F.leaky_relu(v,0.25) for k, v in h.items()}
        h = {k: F.dropout(v,self.dropout, training=self.training) for k, v in h.items()}
        h = self.conv2(graph, h)
        h = {k: F.leaky_relu(v,0.25) for k, v in h.items()}
        h = {k: F.dropout(v, self.dropout, training=self.training) for k, v in h.items()}
        h = self.conv3(graph, h)
        h = {k: F.leaky_relu(v,0.25) for k, v in h.items()}
        h = {k: F.dropout(v, self.dropout, training=self.training) for k, v in h.items()}
        return h

class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x


class TCP(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_class, dropout):
        super().__init__()
        self.views = len(in_dim)
        self.classes = num_class
        self.dropout = dropout

        self.TCPConfidenceLayer = nn.ModuleList([LinearLayer(hidden_dim[0], 1) for _ in range(self.views)])
        self.TCPClassifierLayer = nn.ModuleList([LinearLayer(hidden_dim[0], num_class) for _ in range(self.views)])
        self.selfattentionLayer=Self_attention(hidden_dim[0])

        self.MMClasifier = []
        for layer in range(1, len(hidden_dim) - 1):
            self.MMClasifier.append(LinearLayer(self.views * hidden_dim[0], hidden_dim[layer]))
            self.MMClasifier.append(nn.ReLU())
            self.MMClasifier.append(nn.Dropout(p=dropout))
        if len(self.MMClasifier):
            self.MMClasifier.append(LinearLayer(hidden_dim[-1], num_class))
        else:
            self.MMClasifier.append(LinearLayer(self.views * hidden_dim[-1], num_class))
        self.MMClasifier = nn.Sequential(*self.MMClasifier)

    def forward(self, feature, label=None, infer=False):
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        TCPLogit, TCPConfidence =   dict(), dict()
        all_cord = []
        for view in range(self.views):
            feature[view] = F.relu(feature[view])
            feature[view] = F.dropout(feature[view], self.dropout, training=self.training)
            TCPLogit[view] = self.TCPClassifierLayer[view](feature[view])
            TCPConfidence[view] = self.TCPConfidenceLayer[view](feature[view])
            feature[view] = feature[view] * TCPConfidence[view]
            all_cord.append(feature[view])
        MMfeature = self.selfattentionLayer(all_cord)
        MMlogit = self.MMClasifier(MMfeature)
        if infer:
            return MMlogit
        MMLoss = torch.mean(criterion(MMlogit, label))
        for view in range(self.views):
            pred = F.softmax(TCPLogit[view], dim=1)
            p_target = torch.gather(input=pred, dim=1, index=label.unsqueeze(dim=1)).view(-1)
            confidence_loss = torch.mean(
                F.mse_loss(TCPConfidence[view].view(-1), p_target) + criterion(TCPLogit[view], label))
            MMLoss = MMLoss + confidence_loss
        return MMLoss, MMlogit

    def infer(self, data_list):
        MMlogit = self.forward(data_list, infer=True)
        return MMlogit

class Classifier_1(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x


class Self_attention(nn.Module):
    def __init__(self,dim_he_list):
        super().__init__()
        self.ScaledDotProductAttention=ScaledDotProductAttention(d_model=dim_he_list, d_k=dim_he_list, d_v=dim_he_list, h=8)
        self.ScaledDotProductAttention.apply(xavier_init)
        self.dropout=0.5

    def forward(self, in_list):
        num_view = len(in_list)
        num_sam = in_list[0].shape[0]
        num_feat = in_list[0].shape[1]
        if num_view ==3:
            out_feat =torch.stack([in_list[0], in_list[1],in_list[2]], dim=1)
        if num_view ==2:
            out_feat =torch.stack([in_list[0], in_list[1]], dim=1)
        if num_view == 1:
            out_feat =in_list[0]
        SA_out_feat=self.ScaledDotProductAttention(out_feat,out_feat,out_feat)
        #SA_out_feat=out_feat
        SA_out_feat = F.leaky_relu(SA_out_feat, 0.25)
        SA_out_feat = F.dropout(SA_out_feat, self.dropout, training=self.training)
        SA_out_feat=SA_out_feat.reshape(-1,num_view*num_feat)
        return SA_out_feat



def init_model_dict(num_view, num_class, dim_list,dim_gen_list, out_feat,dim_he_list, rel_names, model_dopout=0.5):
    model_dict = {}
    for i in range(num_view):
        model_dict["E{:}".format(i + 1)] = HGCN([dim_list[i],dim_gen_list[i]],out_feat, dim_he_list, model_dopout,rel_names)
        model_dict["C{:}".format(i+1)] = Classifier_1(dim_he_list[-1], num_class)
    model_dict["Fus"]=TCP(dim_list,[dim_he_list[-1]],num_class,model_dopout)
    return model_dict


def init_optim(num_view, model_dict, lr_e=1e-4):
    optim_dict = {}
    for i in range(num_view):
        optim_dict["C{:}".format(i+1)] = torch.optim.Adam(
                list(model_dict["E{:}".format(i+1)].parameters())+list(model_dict["C{:}".format(i+1)].parameters()),
                lr=lr_e)
    optim_dict["M"] = torch.optim.Adam(list(model_dict["Fus"].parameters()),lr=lr_e)

    return optim_dict