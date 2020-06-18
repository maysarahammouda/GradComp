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


        a = torch.cat((inputs,inputs),0)
        print("in", a)
