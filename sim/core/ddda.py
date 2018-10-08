import numpy as np

class Statistic:
    # Initialise a new Statistic Object
    def __init__(self):
        self.NSamples   = 0.0
        self.Sum        = 0.0
        self.SumSq      = 0.0

    # Add new data element to given Statistic object
    def addData(self, new_sample):
        self.NSamples   += 1
        self.Sum        += new_sample
        self.SumSq      += new_sample*new_sample

    # Add two statistic objects and return the result
    def addition(A, B):
        C = Statistic()
        C.NSamples   = A.NSamples + B.NSamples
        C.Sum        = A.Sum + B.Sum
        C.SumSq      = A.SumSq + B.SumSq
        return C

    def mean(self):
        return self.Sum/self.NSamples

    def var(self):
        return (self.SumSq / self.NSamples + self.mean() ** 2) / (self.NSamples - 1)

    def sigma(self):
        return np.sqrt(self.varience())

    def chi(self):
        return self.error() * np.sqrt(2.0/ (self.NSamples-1))

class Decorrelation:
    # Initialise a new Decorrelation Object
    def __init__(self):
        self.Size = 0
        self.NSamples = 0
        self.BlockedDataStatistics = [Statistic()]
        self.waiting_sample = [0]
        self.waiting_sample_exists = [False]

    def addData(self, new_sample):
        # Add the new sample to the unblocked Statistic
        self.BlockedDataStatistics[0].addData(new_sample)

        # Increment the total number of samples
        self.NSamples += 1

        # Create a new data block when passing threshold
        if self.NSamples >= 2**(self.Size):
            self.Size += 1
            self.BlockedDataStatistics.append(Statistic())
            self.waiting_sample.append(0)
            self.waiting_sample_exists.append(False)

        # Propagate the new sample up through the data structure
        done = False
        carry = new_sample
        i = 1
        while (not done):
            if self.waiting_sample_exists[i]:
                new_sample = (self.waiting_sample[i] + carry)/2
                carry = new_sample
                self.BlockedDataStatistics[i].addData(new_sample)
                self.waiting_sample_exists[i] = False
            else:
                self.waiting_sample_exists[i] = True
                self.waiting_sample[i] = carry
                done = True
            i = i+1
            if i > self.Size:
                done = True


    # Add two Decorrelation objects and return the result

    def addition(A,B):
        C = Decorrelation()
        C.NSamples = A.NSamples + B.NSamples

        # Make C big enough to hold all the data from A and B

        while C.NSamples >= 2**C.Size:

            C.Size = C.Size + 1
            C.BlockedDataStatistics = C.BlockedDataStatistics.append(Statistic())
            C.waiting_sample = C.waiting_sample.append(0)
            C.waiting_sample_exists = C.waiting_sample_exists.append(False)
            carry_exists = False
            carry = 0

            for i in range(C.size):

                if i <= A.Size:

                    StatA = A.BlockedDataStatistics[i]
                    waiting_sampleA = A.waiting_sample[i]
                    waiting_sample_existsA = A.waiting_sample_exists[i]

                else:

                    StatA = Statistic()
                    waiting_sampleA = 0
                    waiting_sample_existsA = False

                if i <= B.Size:

                    StatB = B.BlockedDataStatistics[i]
                    waiting_sampleB = B.waiting_sample[i]
                    waiting_sample_existsB = B.waiting_sample_exists[i]

                else:
                    StatB = Statistic()
                    waiting_sampleA = 0
                    waiting_sample_existsA = False
                    C.BlockedDataStatistics[i] = C.BlockedDataStatistics[i].addition(StatA, StatB)

                if (carry_exists and waiting_sample_existsA and waiting_sample_existsB):
                    # Three samples to handle
                    C.BlockedDataStatistics[i].addData((waiting_sampleA+waiting_sampleB)/2)
                    C.waiting_sample[i] = carry
                    C.waiting_sample_exists[i] = True
                    carry_exists = True
                    carry = (waiting_sampleA+waiting_sampleB)/2

                elif (not carry_exists and waiting_sample_existsA and waiting_sample_existsB):
                    # Two samples to handle
                    C. BlockedDataStatistics[i].addData((waiting_sampleA+waiting_sampleB)/2)
                    C.waiting_sample[i] = 0
                    C.waiting_sample_exists[i] = False
                    carry_exists = True
                    carry = (waiting_sampleA+waiting_sampleB)/2

                elif (carry_exists and not waiting_sample_existsA and waiting_sample_existsB):
                    # Two samples to handle
                    C.BlockedDataStatistics[i].addData((carry+waiting_sampleB)/2)
                    C.waiting_sample[i] = 0
                    C.waiting_sample_exists[i] = False
                    carry_exists = True
                    carry = (carry+waiting_sampleB)/2

                elif (carry_exists and waiting_sample_existsA and not waiting_sample_existsB):
                    # Two samples to handle
                    C.BlockedDataStatistics[i].addData((carry+waiting_sampleA)/2)
                    C.waiting_sample[i] = 0
                    C.waiting_sample_exists[i] = False
                    carry_exists = True
                    carry = (carry+waiting_sampleA)/2

                elif (carry_exists or waiting_sample_existsA or waiting_sample_existsB):
                    # One sample to handle
                    C.waiting_sample[i] = carry + waiting_sampleA + waiting_sampleB
                    C.waiting_sample_exists[i] = True
                    carry_exists = False
                    carry = 0

                else:
                    # No samples to handle
                    C.waiting_sample[i] = 0
                    C.waiting_sample_exists[i] = False
                    carry_exists = False
                    carry = 0
        return C

A = Decorrelation()
A.addData(np.array([1, 0]))#
print(A.BlockedDataStatistics[0].mean())

A.addData(np.array([2, 1]))
print(A.BlockedDataStatistics[0].mean())
print(A.BlockedDataStatistics[1].mean())

A.addData(np.array([2, 1]))
print(A.BlockedDataStatistics[0].mean())
print(A.BlockedDataStatistics[1].mean())

A.addData(np.array([3, 2]))
print(A.BlockedDataStatistics[0].mean())
print(A.BlockedDataStatistics[1].mean())

A.addData(np.array([5, 2]))
print(A.BlockedDataStatistics[0].mean())
print(A.BlockedDataStatistics[1].mean())
print(A.waiting_sample[1])
# print(A.BlockedDataStatistics[2].mean())
# print(A.BlockedDataStatistics[1].mean())
print(len(A.BlockedDataStatistics))

