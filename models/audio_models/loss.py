import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropy(nn.Module):
    def __init__(self, embedding_size, num_classes):
        super(CrossEntropy, self).__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.fc = nn.Linear(embedding_size, num_classes)

    def forward(self, embeddings, labels):
        logits = self.fc(embeddings)
        loss = F.cross_entropy(logits + 1e-8, labels)
        return loss, logits

class OnlineTriplet(nn.Module):
    def __init__(self, margin, triplet_selector):
        super(OnlineTriplet, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, labels):
        triplets = self.triplet_selector.get_triplets(embeddings, labels)
        if embeddings.is_cuda:
            triplets = triplets.cuda()
        ap_cosine_score = F.cosine_similarity(embeddings[triplets[:,0]], embeddings[triplets[:,1]])
        an_cosine_score = F.cosine_similarity(embeddings[triplets[:,0]], embeddings[triplets[:,2]])
        losses = F.relu(an_cosine_score - ap_cosine_score + self.margin)
        return losses.mean(), len(triplets)

class LMCL(nn.Module):
    def __init__(self, embedding_size, num_classes, s, margin):
        super(LMCL, self).__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.s = s
        self.margin = margin
        self.weights = nn.Parameter(torch.Tensor(num_classes, embedding_size))
        nn.init.kaiming_normal_(self.weights)

    def forward(self, embeddings, labels):
        logits = F.linear(F.normalize(embeddings), F.normalize(self.weights))
        margin = torch.zeros_like(logits)
        margin.scatter_(1, labels.view(-1,1), self.margin)
        m_logits = self.s * (logits - margin)
        loss = F.cross_entropy(m_logits + 1e-8, labels)
        # L1 norm
        loss += 0.00001 * torch.norm(self.weights, 1)
        return loss, logits

# TODO
class ASoftmax(nn.Module):
    def __init__(self):
        super(ASoftmax, self).__init__()

    def forward(self, x):
        pass
    
# TODO
class AAMSoftmax(nn.Module):
    def __init__(self):
        super(AAMSoftmax, self).__init__()

    def forward(self, x):
        pass

# TODO
class Contrastive(nn.Module):
    def __init__(self):
        super(Contrastive, self).__init__()

    def forward(self, x):
        pass

if __name__ == '__main__':
    criterion = CrossEntropy(512, 1211)
    inputs = torch.randn(32, 512)
    labels = torch.randint(0, 1211, (32,))
    output, logits = criterion(inputs, labels)
    print(output)
    cross_entropy = nn.CrossEntropyLoss()
    loss = cross_entropy(logits, labels)
    print(loss)
    print(F.softmax(logits, dim = 1).sum(dim = 1, keepdim = True))
