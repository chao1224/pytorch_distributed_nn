mpirun -n 3 --hostfile hosts_address python distributed_nn.py --network=ResNet18 --dataset=Cifar10 --batch-size=1024 --comm-type=Bcast
