if (((batch + 1) % args.num_workers == 0 and batch > 0) or (batch == num_seq - 1)):



# for param in model.parameters():
      #     grad.append(param.grad.view(-1))
      # print ("grad length:", len(grad))
      # print (grad[0:3])
      # print("grad[0]:", grad[0])
      # print ("grad[1]:", grad[1])
      # print("grad[1].shape:", grad[1].shape)
      # print("grad[10].shape:", grad[10].shape)
      # print ("grad[0]+grad[1]:", grad[0]+grad[1])
      # print("(grad[0]+grad[1]).shape:", (grad[0]+grad[1]).shape)

# to count how many non-zero elements we have
print((comp_grads != 0).sum(dim=0))


        a = torch.cat((inputs,inputs),0)
        print("in", a)

# if 't0' in optimizer.param_groups[0]:
#     tmp = {}
#     for prm in model.parameters():
#         tmp[prm] = prm.data.clone()
#         prm.data = optimizer.state[prm]['ax'].clone()

# for prm in model.parameters():
#     prm.data = tmp[prm].clone()


    # print('number of full sequence', num_fullSeq)
    # print('last sequence length', last_seqLen)
    # print('number of sequence', num_seq)
    # print('last update worker', last_update_worker)

def generate_batch2(raw_data, batch_size, num_steps, num_workers):
    """
    This function iterates on the raw data and generates batch_size pointers,
    which allows minibatch iteration along these pointers.
    Args:
        raw_data: one of the raw data outputs from the raw_data function:
                train_data, valid_data, or test_data.
        batch_size: the batch size (int).
        num_steps: the number of unrolls (int).
    Yields:
        Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
        The second element of the tuple is the same data time-shifted to the right
        by one.
    Raises:
        ValueError: if batch_size or num_steps are too high.
    """
    raw_data = np.array(raw_data, dtype=np.int32)
    print("raw_data:", raw_data)

    data_len = len(raw_data)
    print("len of raw data:", data_len)

    eff_batch_size = batch_size * num_workers
    batch_len = data_len // eff_batch_size
    print("batch_len:", batch_len)
    print("batch_size:", batch_size)
    print("num_steps:",num_steps)

    data = np.zeros([eff_batch_size*num_steps, batch_len//num_steps], dtype=np.int32)
    print("data.shape:", data.shape)

    for i in range(eff_batch_size*num_steps):
        data[i] = raw_data[batch_len // num_steps * i:batch_len // num_steps * (i + 1)]
    print("data:",data)

    epoch_size = (batch_len - 1) // num_steps
    print("Epoch Size= ", epoch_size)

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        for j in range(num_workers):
            x = data[j*batch_size:(j+1)*batch_size, i*num_steps:(i+1)*num_steps]
            # print("x", x)
            # print(type(x))
            # print(x.shape)
            y = data[j*batch_size:(j+1)*batch_size, i*num_steps+1:(i+1)*num_steps+1]
            # print("y", y)
            # print(type(y))
            # print(y.shape)

            yield (x, y)


def generate_batch(raw_data, batch_size, num_steps, num_workers):
    """
    This function iterates on the raw data and generates batch_size pointers,
    which allows minibatch iteration along these pointers.
    Args:
        raw_data: one of the raw data outputs from the raw_data function:
                train_data, valid_data, or test_data.
        batch_size: the batch size (int).
        num_steps: the number of unrolls (int).
    Yields:
        Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
        The second element of the tuple is the same data time-shifted to the right
        by one.
    Raises:
        ValueError: if batch_size or num_steps are too high.
    """
    raw_data = np.array(raw_data, dtype=np.int32)
    print("raw_data:", raw_data)

    data_len = len(raw_data)
    print("len of raw data:", data_len)

    eff_batch_size = batch_size * num_workers
    batch_len = data_len // eff_batch_size
    print("batch_len:", batch_len)
    print("batch_size:", batch_size)
    print("num_steps:",num_steps)

    data = np.zeros([eff_batch_size, batch_len], dtype=np.int32)

    for i in range(eff_batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]
    print(data)
    # data = np.zeros([batch_len//num_steps, eff_batch_size*num_steps], dtype=np.int32)
    #
    # for i in range(batch_len//num_steps):
    #     data[i] = raw_data[eff_batch_size*num_steps * i:eff_batch_size*num_steps * (i + 1)]
    # print(data)

    epoch_size = (batch_len - 1) // num_steps
    print("Epoch Size= ", epoch_size)

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        for j in range(num_workers):
            x = data[j*batch_size:(j+1)*batch_size, i*num_steps:(i+1)*num_steps]
            # x = data[i:(i+1), j*batch_size*num_steps:(j+1)*batch_size*num_steps]
            print("x", x)
            # x = np.reshape(x,(num_steps, eff_batch_size))
            # print("x", x)
            # print(type(x))
            print(x.shape)
            y = data[j*batch_size:(j+1)*batch_size, i*num_steps+1:(i+1)*num_steps+1]
            # y = data[i+1:(i+1)+1, j*batch_size*num_steps:(j+1)*batch_size*num_steps]
            print("y", y)
            # y = np.reshape(y,(num_steps, eff_batch_size))
            # print("y", y)
            # print(type(y))
            print(y.shape)

            yield (x, y)

def batch_generator(raw_data, batch_size, num_steps):
    """
    This function iterates on the raw data and generates batch_size pointers,
    which allows minibatch iteration along these pointers.
    Args:
        raw_data: one of the raw data outputs from the raw_data function:
                train_data, valid_data, or test_data.
        batch_size: the batch size (int).
        num_steps: the number of unrolls (int).
    Yields:
        Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
        The second element of the tuple is the same data time-shifted to the right
        by one.
    Raises:
        ValueError: if batch_size or num_steps are too high.
    """
    raw_data = np.array(raw_data, dtype=np.int32)
    print("raw_data:", raw_data)

    data_len = len(raw_data)
    print("len of raw data:", data_len)
    batch_len = data_len // batch_size
    print("batch_len:", batch_len)
    print("batch_size:", batch_size)
    print("num_steps:",num_steps)

    data = np.zeros([batch_size, batch_len], dtype=np.int32)

    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]
    print(data)

    epoch_size = (batch_len - 1) // num_steps
    print("Epoch Size= ", epoch_size)

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i*num_steps:(i+1)*num_steps]
        # print("x", x)
        # print(x.shape)
        y = data[:, i*num_steps+1:(i+1)*num_steps+1]
        # print("y", y)
        # print(y.shape)

        yield (x, y)
