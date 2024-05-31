import jax
import jax.numpy as jnp
from torch.utils.data import DataLoader, random_split
from dataset import SpotifyGestureDataset, get_dataset_from_cfg
from mlp import mlp_forward, mlp_serialize_binary, batch_mlp_forward, get_mlp_from_cfg, load_cfg, MLP_config, \
    save_cfg
import optax
import argparse
import tqdm
import wandb
import pickle

def main(args):
    sizes = list(map(int, args.sizes.split(',')))
    key = jax.random.PRNGKey(0)

    dataset: SpotifyGestureDataset
    if args.load_path is not None:
        cfg = load_cfg(args.load_path + '.cfg')
        with(open(args.load_path + '.pkl', 'rb')) as file:
            params = pickle.load(file)
    else:
        cfg = MLP_config(name = args.name, sizes = sizes, modality = args.modality, c = args.c, h = args.h, w = args.w, image_size = args.image_size, classes = args.classes)
        params = get_mlp_from_cfg(cfg, key)
        save_cfg(cfg, f"{args.directory}{args.name}.cfg")

    dataset = get_dataset_from_cfg(args.data_path, cfg)

    solver = optax.adamw(learning_rate=1e-3)
    opt_state = solver.init(params)

    trainset, testset = random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    wandb.init(project="spotify_gesture", name=args.name)

    train_loop(args, params, train_loader, test_loader, opt_state, solver)

def train_loop(args, params, train_loader, test_loader, opt_state, solver):
    """ Training loop """
    for epoch in range(args.epochs):
        logger_dict = {}
        for batch_num, (inputs, labels) in tqdm(enumerate(train_loader)):
            train_loss, grads = forward_backward(params, inputs, labels)
            updates, opt_state = solver.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            
            logger_dict['loss'] = train_loss.item()
            logger_dict['epoch'] = epoch
            tqdm.set_postfix(loss=train_loss.item())

        for batch_num, (inputs, labels) in tqdm(enumerate(test_loader)):
            test_loss, _ = forward_backward(params, inputs, labels)
            logger_dict['loss'] = test_loss.item()
            logger_dict['epoch'] = epoch
            tqdm.set_postfix(loss=test_loss.item())

        if epoch % args.save_interval == 0:
            mlp_serialize_binary(params, f"{args.name}_{epoch}.bin")
            with open(f"{args.directory}{args.name}_{epoch}.pkl", 'wb') as file:
                pickle.dump(params, file)

        if epoch % args.sanity_interval == 0:
            inputs, labels = next(iter(test_loader))

            inputs = inputs[0].squeeze(0)
            labels = labels[0].squeeze(0)
            logits = mlp_forward(params, inputs)
            pred = jnp.argmax(logits)
            logger_dict['pred'] = pred
            logger_dict['true'] = labels
            wandb.log({"inputs": [wandb.Image(inputs[0]), wandb.Image(inputs[1])]})


        wandb.log(logger_dict)

        return params, opt_state

@jax.value_and_grad
def forward_backward(params, inputs, labels):
    """ Cross entropy loss """
    logits = batch_mlp_forward(params, inputs)
    return -jnp.mean(jax.nn.log_softmax(logits) * labels)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data_path', type=str, default='path/to/dataset')
    parser.add_argument('--sizes', type=str, default='512,256')
    parser.add_argument('--classes', type=int, default=3)
    parser.add_argument('-m', '--modality', type=str, default='RGB')
    parser.add_argument('--c', type=int, default=3)
    parser.add_argument('--h', type=int, default=256)
    parser.add_argument('--w', type=int, default=256)
    parser.add_argument('-i', '--image_size', type=int, default=2048)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_arugment('--load_path', type=str, default=None)
    parser.add_argument('--save_interval', type=int, default=25)
    parser.add_argument('--sanity_interval', type=int, default=25)
    parser.add_argument('--name', type=str, default='model')
    parser.add_argument("-d", "--directory", type=str, default="weights/")
    
    args = parser.parse_args()
    main(args)