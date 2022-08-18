import time
t = time.time()
import argparse
import torch
print("CUDA Available: " + str(torch.cuda.is_available()))
import torch.nn as nn
from Transformer import *
from jointEmbedding import JointEmbeddingModel
from HelperFunctions import *
from os.path import exists
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    je_model = JointEmbeddingModel(args.embed_size).to(device)
    params = list(je_model.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=0.000001)
    tokenizer = Report_Tokenizer()

    mp = args.model_path
    mp += 'CLIP_model'

    if hasattr(args, 'synthetic') and args.synthetic:
        mp = mp + 'synth/'
    print(mp)

    fp = getExperiment(args, mp)
    print("experiment path", fp)
    start, best_val_loss, args = startExperiment(args, je_model, optimizer, fp)

    filts = 'frontal'
    if exists(fp + '/filters.txt'):
        filters = getFilters(fp)
        priors = set(getFilters(fp, overwrite= filts, toprint=False))
        if set(filters) != priors:
            print("Warning: entered filters differ from those previously used: " + priors)
    else:
        filters = getFilters(fp, overwrite = filts)
        if fp != 'debug':
            with open(fp + '/filters.txt', 'w') as f:
                f.write(filts)

    # Build data
    if args.debug:
        subset = ['tinytrain', 'tinyval']
    else:
        subset = ['train', 'val']

    mimic_dat = getDatasets(subset = subset, augs = 2, filters = filters)
    train_data_loader_mimic, val_data_loader_mimic = getLoaders(mimic_dat, args, subset=subset, num_work=16)
    total_step_mimic = len(train_data_loader_mimic)

    for epoch in range(start, args.num_epochs):
        je_model.train()
        tmimic = time.time()
        for i, (im1, im2, df, texts) in enumerate(train_data_loader_mimic):
            loss = train(je_model, im1, im2, texts, tokenizer)
            loss.backward()
            optimizer.step()
            if i % args.log_step == 0:
                print('MIMIC Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch, args.num_epochs, i, total_step_mimic, loss.item()))
        print("Mimic Epoch time: " + str(time.time() - tmimic))

        if epoch % args.val_step == 0:
            print("Validating/saving model")
            je_model.eval()
            tval = time.time()
            val_loss = validate(val_data_loader_mimic, tokenizer, je_model)
            if not args.debug:
                torch.save({'epoch': epoch+1,
                            'model_state_dict': je_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_loss': val_loss,
                            'best_val_loss':best_val_loss,
                            'args': args}, os.path.join(fp, 'je_model-{}.pt'.format(epoch)))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({'epoch': epoch + 1,
                                'model_state_dict': je_model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'val_loss': val_loss,
                                'best_val_loss':best_val_loss,
                                'args': args}, os.path.join(fp, 'best_model.pt'.format(epoch)))

            print("Val time " + str(time.time() - tval))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/n/data2/hms/dbmi/beamlab/anil/Conformal_CLIP/models/', help='path for saving trained models')
    parser.add_argument('--log_step', type=int, default=500, help='step size for printing log info')
    parser.add_argument('--val_step', type=int, default=2, help='step size for printing val info')
    parser.add_argument('--resume', type=bool, default=False, const=True, nargs='?')
    parser.add_argument('--desc', type=str, default="", help='experiment description')
    parser.add_argument('--debug', type=bool, default=False, const = True, nargs='?', help='debug mode, dont save')

    # Model parameters
    parser.add_argument('--embed_size', type=int, default=128, help='dimension of embeddings')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=.0001)
    args = parser.parse_args()
    print(args)
    main(args)