import jax
import jax.numpy as jnp
from torch.utils.data import DataLoader, random_split
from dataset import SpotifyGestureDataset, get_dataset_from_cfg
from mlp import mlp_forward, mlp_serialize_binary, batch_mlp_forward, get_mlp_from_cfg, load_cfg, MLP_config, \
    save_cfg
import optax
import argparse
from tqdm import tqdm
import wandb
import pickle

def main(args):
    sizes = list(map(int, args.sizes.split(',')))
    key = jax.random.PRNGKey(0)
    model_key, key = jax.random.split(key)

    dataset: SpotifyGestureDataset
    if args.load_path is not None:
        cfg = load_cfg(args.load_path + '.cfg')
        with(open(args.load_path + '.pkl', 'rb')) as file:
            params = pickle.load(file)
    else:
        cfg = MLP_config(name = args.name, sizes = sizes, modality = args.modality, c = args.c, h = args.h, w = args.w, image_size = args.image_size, classes = args.classes)
        params = get_mlp_from_cfg(cfg, model_key)
        save_cfg(cfg, f"{args.directory}{args.name}.cfg")

    dataset = get_dataset_from_cfg(args.data_path, cfg)

    solver = optax.adamw(learning_rate=1e-5)
    opt_state = solver.init(params)

    trainset, testset = random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    wandb.init(project="spotify_gesture", name=args.name)

    key, train_key = jax.random.split(key)
    train_loop(args, params, train_loader, test_loader, opt_state, solver, train_key)

def train_loop(args, params, train_loader, test_loader, opt_state, solver, key):
    """ Training loop """
    for epoch in range(args.epochs):
        logger_dict = {}
        # Instantiate the tqdm progress bar around the enumerator of train_loader
        with tqdm(enumerate(train_loader), total=len(train_loader), leave=False, desc=f"Training Epoch {epoch}") as pbar:
            for batch_num, (inputs, labels) in pbar:
                inputs = jnp.array(inputs)  # Convert inputs to JAX's numpy array
                labels = jnp.array(labels)  # Convert labels to JAX's numpy array
                labels = jax.nn.one_hot(labels, args.classes, dtype=jnp.float32)  # One-hot encode labels

                train_loss, grads = forward_backward(params, inputs, labels)  # Compute loss and gradients
                updates, opt_state = solver.update(grads, opt_state, params)  # Update parameters
                params = optax.apply_updates(params, updates)  # Apply updates

                logger_dict['loss'] = train_loss.item()
                logger_dict['epoch'] = epoch

                # Update tqdm postfix to display loss information
                pbar.set_postfix(loss=train_loss.item(), epoch=epoch)
        
            pbar.close()

        with tqdm(enumerate(test_loader), total=len(test_loader), leave=False, desc=f"Val Epoch {epoch}") as pbar:
            for batch_num, (inputs, labels) in pbar:
                inputs = jnp.array(inputs)
                labels = jnp.array(labels)
                labels = jax.nn.one_hot(labels, args.classes)

                test_loss, _ = forward_backward(params, inputs, labels)
                logger_dict['val_loss'] = test_loss.item()
                pbar.set_postfix(loss=test_loss.item())

            pbar.close()
        
        if epoch % args.val_accuracy_interval == 0:
            correct = 0
            total = 0
            for inputs, labels in test_loader:
                inputs = jnp.array(inputs)
                labels = jnp.array(labels)
                labels = jax.nn.one_hot(labels, args.classes)

                logits = batch_mlp_forward(params, inputs)
                pred = jnp.argmax(logits, axis=1)
                correct += jnp.sum(pred == jnp.argmax(labels, axis=1))
                total += inputs.shape[0]
            val_accuracy = correct / total
            logger_dict['val_accuracy'] = val_accuracy

        if epoch % args.save_interval == 0:
            mlp_serialize_binary(params, f"{args.directory}{args.name}_{epoch}.bin")
            with open(f"{args.directory}{args.name}_{epoch}.pkl", 'wb') as file:
                pickle.dump(params, file)

        if epoch % args.sanity_interval == 0:
            inputs, labels = next(iter(train_loader))

            key, idx_key = jax.random.split(key)
            idx = jax.random.randint(idx_key, (1,), 0, args.batch_size).item()
            inputs = inputs[idx].squeeze(0)
            labels = labels[idx].squeeze(0)
            gt = labels
            inputs = jnp.array(inputs)
            labels = jnp.array(labels)
            labels = jax.nn.one_hot(labels, args.classes)
            logits = mlp_forward(params, inputs)
            pred = jnp.argmax(logits)
            logger_dict['pred'] = pred
            logger_dict['gt'] = gt
            # wandb.log({"inputs": [wandb.Image(inputs[0]), wandb.Image(inputs[1])]})


        wandb.log(logger_dict)

    return params, opt_state

@jax.jit
@jax.value_and_grad
def forward_backward(params, inputs, labels):
    """ Cross entropy loss """
    logits = batch_mlp_forward(params, inputs)
    return -jnp.mean(jax.nn.log_softmax(logits) * labels)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data_path', type=str, default='../test_data/')
    parser.add_argument('--sizes', type=str, default='512,256')
    parser.add_argument('--classes', type=int, default=3)
    parser.add_argument('-m', '--modality', type=str, default='JPEG')
    parser.add_argument('--c', type=int, default=3)
    parser.add_argument('--h', type=int, default=256)
    parser.add_argument('--w', type=int, default=256)
    parser.add_argument('-i', '--image_size', type=int, default=2200)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--save_interval', type=int, default=50)
    parser.add_argument('--sanity_interval', type=int, default=5)
    parser.add_argument('--val_accuracy_interval', type=int, default=5)
    parser.add_argument('--name', type=str, default='big_test')
    parser.add_argument("-d", "--directory", type=str, default="../weights/")
    
    args = parser.parse_args()
    main(args)