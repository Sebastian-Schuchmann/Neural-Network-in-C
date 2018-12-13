using System;
using System.Collections.Generic;
using Accord;
using Accord.Math;

namespace NeuralNetwork
{
    class Program
    {
        static void Main(string[] args)
        {
            var TrainingData = MNIST.ReadMNIST(false);
            //Set Random Seed for better debugging
            Accord.Math.Random.Generator.Seed = 5;
            //Input, Hidden, Output
            NeuralNet NN = new NeuralNet(2, 2, 1);
            TrainXORTest(NN);
        }

        private static void TrainXORTest(NeuralNet NN)
        {
            double[][] TrainData = {
                    new double[]{0, 1},
                    new double[]{1, 1},
                    new double[]{1, 0},
                    new double[]{0, 0},
            };

            var Labels = new double[][]
                {
                    new double[]{0},
                    new double[]{1},
                    new double[]{0},
                    new double[]{0},
            };

            for (int i = 1; i < 10000; i++)
            {
                TrainData.Shuffle();
                NN.Train(TrainData, Labels, 0.01);
            }
        }
    }

    class NeuralNet
    {
        double[,] WeightsInputHidden;
        double[,] WeightsHiddenOutput;

        double[] Inputs;
        double[] Hidden;
        double[] BiasHidden;
        double[] BiasOutput;
        double[] Output;

        public NeuralNet(int AmountInputs, int AmountHidden, int AmountOutput)
        {
            //Weight Matrices
            WeightsInputHidden = Matrix.Random(AmountHidden, AmountInputs);
            WeightsHiddenOutput = Matrix.Random(AmountOutput, AmountHidden);

            //Vectors for Input Output and Hidden
            Inputs = Vector.Create(AmountInputs, 0.0);
            //Each Output/Hidden Node also has a Bias
            Hidden = Vector.Create(AmountHidden, 0.5);
            BiasHidden = Vector.Create(AmountHidden, 0.5);
            Output = Vector.Create(AmountOutput, 0.0);
            BiasOutput = Vector.Create(AmountOutput, 0.5);

            Visualize();
        }

        public (double[] output, double[] weightedSumHidden, double[] weightedSumOutput) Fetch(double[] FetchInputs)
        {
            if (FetchInputs.Length != Inputs.Length)
            {
                throw new Exception("Input Shape doesnt match!");
            }

            SetInputs(FetchInputs);

            //Calculate weighted sums of hidden nodes
            Hidden = WeightsInputHidden.Dot(Inputs);
            Hidden = Hidden.Add(BiasHidden);
            //Activation function
            var Hidden_NoActivation = Hidden;
            Hidden = ApplyReLU(Hidden);

            Output = WeightsHiddenOutput.Dot(Hidden);
            Output.Add(BiasOutput);
            var Output_NoActivation = Output;
            Output = ApplyReLU(Output);

            VisualizeIO();
            return (Output, Hidden_NoActivation, Output_NoActivation);
        }

        public void Train(double[][] data, double[][] labels, double learningRate)
        {
            for (int i = 0; i < data.Length; i++){
                var feedforward = Fetch(data[i]);
                var output = feedforward.output;
                var weightedSumOutput = feedforward.weightedSumOutput;

                var Labels = new double[][]
                {
                    new double[]{0},
                };

                Labels[0][0] = GenerateXORLabel(data[i][0], data[i][1]);



                //This is basically the cost function
                var errors_output = Labels[0].Subtract(output); //How good is to output compared to the target

                /*---------------------------------------------------------------
                 * -- Calculate Delta of the weight between hidden and output --
                 ---------------------------------------------------------------*/
                var HiddenTransposed = Hidden.Transpose();
                var deltaWeightOutput = HiddenTransposed.Dot(errors_output);
                double[,] deltaWeightOutput2D = Matrix.Create(deltaWeightOutput); //Convert to Matrix

                /*---------------------------------------------------------------
                 * -- Calculate Delta of the weight between input and hidden --
                 ---------------------------------------------------------------*/
                //First we have to calculate the Error in the hidden nodes ...
                //Transposed because we are going Backwards through the Network
                var WHOTransposed = WeightsHiddenOutput.Transpose();
                //Moves the Error to the output layer
                var errors_hidden = WHOTransposed.Dot(errors_output);
                //Element Wise multiplication (schur product)
                var weightedSumHidden = feedforward.weightedSumHidden;
                weightedSumHidden = ApplyDerivativeReLU(weightedSumHidden);
                //Moves the Error backthrough the Neuron
                errors_hidden = errors_hidden.Multiply(weightedSumHidden);

                //... then we can Calculate the Delta
                var InputTransposed = Inputs.Transpose();
                var deltaWeightHidden = InputTransposed.Dot(errors_hidden);
                double[,] deltaWeightHidden2D = Matrix.Create(deltaWeightHidden); //Convert to Matrix
                deltaWeightHidden2D = Inputs.Transpose().Dot(deltaWeightHidden2D);
          
                /*---------------------------------------------------------------
                 * --        Adjust Weights and Biases using the delta         --
                 ---------------------------------------------------------------*/
                //The Biases just get adjusted by adding the Errors multiplied by the learning rate
                BiasOutput = BiasOutput.Add(errors_output.Multiply(learningRate)); //Output Bias
                WeightsHiddenOutput =  WeightsHiddenOutput.Add(deltaWeightOutput2D.Multiply(learningRate));
                BiasHidden = BiasHidden.Add(errors_hidden.Multiply(learningRate)); //Hidden Bias
                WeightsInputHidden = WeightsInputHidden.Add(deltaWeightHidden2D.Multiply(learningRate));
            }

        }

        private double GenerateXORLabel(double v1, double v2)
        {
            if ((v1 > 0.0)^(v2 > 0.0))
            {
                return 1.0;
            }
            return 0.0;
        }

        public void SetWeightsAndBiases(double[,] NewWeightsInput, double[,] NewWeightsOutput,
                                        double[] NewBiasHidden, double[] NewBiasOutput){
            WeightsInputHidden = NewWeightsInput;
            WeightsHiddenOutput = NewWeightsOutput;
            BiasHidden = NewBiasHidden;
            BiasOutput = NewBiasOutput;
        }

        double[] ApplyReLU(double[] set){
            for (int i = 0; i < set.Length; i++)
                set[i] = ReLU(set[i]);

            return set;
        }

        double[] ApplyDerivativeReLU(double[] set)
        {
            for (int i = 0; i < set.Length; i++)
                set[i] = ReLUDerivative(set[i]);

            return set;
        }

        double[] ApplySoftmax(double[] set){
            double sum = 0.0;
            foreach (double item in set)
                sum += Math.Exp(item);

            for (int i = 0; i < set.Length; i++)
                set[i] = Math.Exp(set[i]) / sum;

            return set;
        }

        double[] ApplySpecialSign(double[] set){
            for (int i = 0; i < set.Length; i++){
                set[i] = set[i] > 0.75 ? 1.0 : 0.0;
            }
            return set;
        }

        private void SetInputs(double[] FetchInputs)
        {
            for (int i = 0; i < FetchInputs.Length; i++)
                Inputs[i] = FetchInputs[i];
        }

        //Rectified Linear Unit
        double ReLU(double x)
        {
            var Relu = Math.Max(0, x);// x < 0 ? 0 : x;
            return Relu;
        }

        double ReLUDerivative(double x)
        {
            if (x > 0)
                return 1.0;
            //if (x <= 0)
                //return -1.0;
            return 0.0;
        }

        public void Visualize()
        {
            Console.WriteLine("--- - - - NEURAL NETWORK - - - ---");
            Console.WriteLine("Inputs:");
            Console.WriteLine(Inputs.ToString<double>());
            Console.WriteLine("WeightsInput:");
            Console.WriteLine(WeightsInputHidden.ToString<double>());
            Console.WriteLine("Hidden (ReLU):");
            Console.WriteLine(Hidden.ToString<double>());
            Console.WriteLine("WeightsOutput:");
            Console.WriteLine(WeightsHiddenOutput.ToString<double>());
            Console.WriteLine("Output (Softmax):");
            Console.WriteLine(Output.ToString<double>());
            Console.WriteLine("--- - - - - - - END - - - - - - ---");
        }

        public void VisualizeIO()
        {
            Console.WriteLine("--- - - - FETCH: - - - ---");
            Console.WriteLine("Inputs:");
            Console.WriteLine(Inputs.ToString<double>());
            Console.WriteLine("Output:");
            Console.WriteLine(Output.ToString<double>());
            Console.WriteLine("--- - - - - - - END - - - - - - ---");
        }
    }
}
