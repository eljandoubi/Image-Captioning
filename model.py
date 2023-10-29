import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(weights='DEFAULT')
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        captions = self.word_embeddings(captions[:,:-1]) 
        inputs = torch.cat((features.unsqueeze(1), captions), dim=1)
        lstm_out, _ = self.lstm(inputs)
        outputs = self.linear(lstm_out)
        return outputs

        
        

    def sample(self, inputs, states=None, max_len=20, num_sample=1000):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "

        scores = torch.zeros((num_sample,1),device=inputs.device)

        predictions = torch.ones((num_sample,max_len+1),
                                 device=inputs.device,dtype=int)

        inputs = torch.repeat_interleave(inputs,num_sample,0)
        
        for i in range(max_len+1):
 
            output, states = self.lstm(inputs,states)
            output = self.linear(output.squeeze(dim = 1))

            probabilities = torch.nn.functional.softmax(output, dim=1)

            predicted_index = torch.multinomial(probabilities, 1)

            predictions[:,i] = predicted_index[:,0]
            
            scores += output.gather(1,predicted_index) 
            
            inputs = self.word_embeddings(predicted_index)   

        best = torch.argmax(scores)
        
        return predictions[best].cpu().numpy().tolist()