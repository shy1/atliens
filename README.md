# atliens

```
    def get_gatecycles(self, bsize, rollstep):
        # steps1 = torch.arange(0, 128 * bsize * self.hidden_size)
        # steps1 = steps1.view(128, bsize, self.hidden_size)

        # vdim = self.hidden_size - 1
        vdim = self.hidden_size
        steps1 = torch.arange(0, 128 * bsize * vdim)
        steps1 = steps1.view(128, bsize, vdim)

        steplist = []
        batchlist16 = []

        for timestep in range(steps1.size(0)):
            distance = (-rollstep * (timestep + 1)) % steps1.size(2)
            if distance == 0:
                steplist.append(steps1[timestep].unsqueeze(0))
            else:
                steplist.append(roll1(steps1[timestep], distance).unsqueeze(0))
        sindexes = torch.cat(steplist, dim=0)

        cycledb16 = sindexes
        # print("Gatecycles:", rollstep, cycledb16.size())
        # import sys
        # sys.exit(-1)
        return cycledb16.cuda()

    def get_outcycles(self, bsize, rollstep):
        # vdim = self.hidden_size - 1
        vdim = self.hidden_size
        steps1 = torch.arange(0, 128 * bsize * vdim)
        steps1 = steps1.view(128, bsize, vdim)

        steplist = []
        batchlist16 = []

        for timestep in range(steps1.size(0)):
            distance = (rollstep * (timestep + 1)) % steps1.size(2)
            if distance == 0:
                steplist.append(steps1[timestep].unsqueeze(0))
            else:
                steplist.append(roll1(steps1[timestep], distance).unsqueeze(0))
        sindexes = torch.cat(steplist, dim=0)

        cycledb16 = sindexes
        # print("Outcycles:", rollstep, cycledb16.size())
        return cycledb16.cuda()
```
