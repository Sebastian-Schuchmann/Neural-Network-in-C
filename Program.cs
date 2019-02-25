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
            Accord.Math.Random.Generator.Seed = 4;

            //Input, Hidden, Output
            //NeuralNet NN = new NeuralNet(28*28, 512, 14);

            NeuralNet NN = new NeuralNet(2, 32, 1);
            //NN.Train(TrainingData, 0.1);
            TrainXORTest(NN);
        }

        private static void TrainXORTest(NeuralNet NN)
        {
            for (int i = 1; i < 2; i++)
            {
                (var TrainData, var Labels) = NN.generateXORData(100000);
                NN.Train(TrainData, Labels, 0.01, 1);
            }
        }

        private double GenerateXORLabel(double v1, double v2)
        {
            if ((v1 > 0.0) ^ (v2 > 0.0))
            {
                return 1.0;
            }
            return 0.0;
        }
    }

    class NeuralNet
    {
        double[,] WeightsInputHidden;
        double[,] WeightsHiddenOutput;

        double[] Inputs;
        double[] Hidden;
        double[] HiddenWithoutRElu;
        double[] BiasHidden;
        double[] BiasOutput;
        double[] Output;

        public NeuralNet(int AmountInputs, int AmountHidden, int AmountOutput)
        {
            
            //Weight Matrices
            WeightsInputHidden = Matrix.Random(AmountHidden, AmountInputs);
            WeightsHiddenOutput = Matrix.Random(AmountOutput, AmountHidden);

            //Vectors for Input Output and Hidden
            Inputs = Vector.Create(AmountInputs, 0.5);
            //Each Output/Hidden Node also has a Bias
            Hidden = Vector.Create(AmountHidden, 0.5);
            HiddenWithoutRElu = Vector.Create(AmountHidden, 0.5);
            BiasHidden = Vector.Create(AmountHidden, 0.5);
            Output = Vector.Create(AmountOutput, 0.5);
            BiasOutput = Vector.Create(AmountOutput, 0.5);

            Visualize();
        }

        public (double[] output, double[] weightedSumHidden, double[] weightedSumOutput) Fetch(double[] FetchInputs, bool visualize)
        {
            if (FetchInputs.Length != Inputs.Length
               )
            {
                throw new Exception("Input Shape doesnt match!");
            }

            SetInputs(FetchInputs);


            //Calculate weighted sums of hidden nodes
            Hidden = WeightsInputHidden.Dot(Inputs);
            NaNCheck("Fetch: Calculate weighted sums of hidden nodes"); 
            Hidden = Hidden.Add(BiasHidden);
            NaNCheck("Fetch: Add Bias"); 



            //Activation function
            var Hidden_NoActivation = Hidden;
            HiddenWithoutRElu = Hidden_NoActivation;
            Hidden = ApplyReLU(Hidden);
            NaNCheck("Fetch: Activation"); 

            Output = WeightsHiddenOutput.Dot(Hidden);
            NaNCheck("Fetch: Output"); 
            Output.Add(BiasOutput);
            NaNCheck("Fetch: Output + bias"); 
            var Output_NoActivation = Output;
            Output = ApplyReLU(Output);
            NaNCheck("Fetch: apply relu to output"); 
           // Output.Add(1e-20);

            if(visualize)
            Visualize();

            return (Output, Hidden_NoActivation, Output_NoActivation);
        }

        public void Train(List<DigitImage> LabeledData, double learningRate)
        {
            double[] errors_output_avg = null;
            double[] weightedSumOutput;
            double[] weightedSumHidden = null;
            Console.WriteLine("Data Length: " + LabeledData.Count);
            //Console.ReadLine();

            for (int i = 0; i < LabeledData.Count; i++)
            {
                bool show = i % 100 == 0 || WeightsInputHidden.HasNaN();
                
                    double[][] data = LabeledData[i].pixelsDbl;
                    //Data gets flattend
                var feedforward = Fetch(data.Flatten(), true);
                    var output = feedforward.output;
                    weightedSumOutput = feedforward.weightedSumOutput;
                    weightedSumHidden = feedforward.weightedSumHidden;

                   // double[][] Labels = ;
                    var Labels = new double[][] { LabeledData[i].labelDbl };

                    errors_output_avg = Labels[0].Subtract(output);
                    Console.WriteLine("Error " + errors_output_avg.ToString<double>());
                    SGD(learningRate, weightedSumHidden, errors_output_avg);
            }
        }

        public void Train(double[][] data, double[][] labels, double learningRate, int miniBatches)
        {
            double[] errors_output_avg = null;
            double[] weightedSumOutput;
            double[] weightedSumHidden = null;
            Console.WriteLine("Data Length: " + data.Length);
            //Console.ReadLine();

            for (int i = 0; i < data.Length/miniBatches; i++)
            {
                for (int j = 0; j < miniBatches; j++)
                {
                    var feedforward = Fetch(data[i+j], true);
                    var output = feedforward.output;
                    weightedSumOutput = feedforward.weightedSumOutput;
                    weightedSumHidden = feedforward.weightedSumHidden;

                    var Labels = new double[][]{new double[]{0}};
                    Labels[0][0] = GenerateXORLabel(data[i+j][0], data[i+j][1]);

                    errors_output_avg = Labels[0].Subtract(output);
                    SGD(learningRate, weightedSumHidden, errors_output_avg);
                }
            }
        }

        private void SGD(double learningRate, double[] weightedSumHidden, double[] errors_output)
        {
            fixWeights();
            NaNCheck("SGD begins");
            /*---------------------------------------------------------------
            * -- Calculate Delta of the weight between hidden and output --
            ---------------------------------------------------------------*/
            var HiddenTransposed = Hidden.Transpose();
            NaNCheck("SGD Transpose Hidden");
            var deltaWeightOutput = HiddenTransposed.Dot(errors_output);
            NaNCheck("SGD Calc delta Weight output");
            double[,] deltaWeightOutput2D = Matrix.Create(deltaWeightOutput); //Convert to Matrix

            Console.WriteLine("Delta Weight Output MATRIX: "
             + deltaWeightOutput2D.ToString<double>()
                             );
            Console.WriteLine("Delta Weight Output NON MATRIX:"
                              + deltaWeightOutput.ToString<double>()
                             );

            Console.WriteLine("Delta Weight Output MATRIX LENGTH: "
                              + deltaWeightOutput2D.Length
                             );

            Console.WriteLine("Delta Weight Output MATRIX TOTAL LENGTH: "
                              + deltaWeightOutput2D.GetTotalLength()
                             );

            Console.WriteLine("Weights H Output MATRIX LENGTH: "
                              + WeightsHiddenOutput.Length
                             );

            Console.WriteLine("Weights H Output MATRIX TOTAL LENGTH: "
                              + WeightsHiddenOutput.GetTotalLength()
                             );
       

            NaNCheck("SGD: Delta Weight to Matrix"); 
            WeightsHiddenOutput = WeightsHiddenOutput.Add(deltaWeightOutput2D.Multiply(learningRate));
            ClampWeightsAndBiases();

            NaNCheck("SGD: Apply!"); 
            /*---------------------------------------------------------------
             * -- Calculate Delta of the weight between input and hidden --
             ---------------------------------------------------------------*/
            //weightsInputHidden += _learningRate * hiddenErrors * hiddenOutputs * (1.0 - hiddenOutputs) * inputSignals.Transpose()
            //First we have to calculate the Error in the hidden nodes ...
            //Transposed because we are going Backwards through the Network
            var WHOTransposed = WeightsHiddenOutput.Transpose();
            Console.WriteLine("WEIGHTS HIDDEN OUTPUT TRANSPOSED:");
            Console.WriteLine(WHOTransposed.ToString<double>());
            //Moves the Error to the output layer
            var errors_hidden = WHOTransposed.Dot(errors_output);
            Console.WriteLine("ERRORS HIDDEN");
            Console.WriteLine(errors_hidden.ToString<double>());
            //Element Wise multiplication (schur product)
            weightedSumHidden = ApplyDerivativeReLU(weightedSumHidden);
            //Moves the Error backthrough the Neuron
            errors_hidden = errors_hidden.Multiply(weightedSumHidden);

            //_weightInputHidden += _learningRate * errors_hidden * inputSignals.Transpose()

            Console.WriteLine("ERRORS HIDDEN MULD BY WSUMHIDDEN");
            Console.WriteLine(errors_hidden.ToString<double>());
            Console.WriteLine("ERRORS HIDDEN LENGTH");
            Console.WriteLine(errors_hidden.Length);

            //... then we can Calculate the Delta

            var InputTransposed = Inputs.Transpose();
            Console.WriteLine("INPUTS TRANSPOSED LENGTH");
            Console.WriteLine(InputTransposed.Length);

            ////_weightInputHidden += _learningRate * errors_hidden * inputSignals.Transpose()
            var error_hidden_m = Matrix.Create(errors_hidden);
            Console.WriteLine("ERRORS HIDDEN MATRIX LENGTH");
            Console.WriteLine(error_hidden_m.Length);
            Console.WriteLine("ERRORS HIDDEN MATRIX");
            Console.WriteLine(error_hidden_m.ToString<double>());
            //var deltaWeightHidden =  InputTransposed.Dot(error_hidden_m);
            var InputMatrix = Matrix.Create(Inputs);
            InputMatrix = InputMatrix.Transpose();
            var deltaWeightHidden = InputMatrix.Dot(errors_hidden);
            Console.WriteLine("deltaWeightHidden MATRIX");
            Console.WriteLine(deltaWeightHidden.ToString<double>());



            Console.WriteLine("FIRST STEP DELTA WEIGHT HIDDEN LENGTH");
            Console.WriteLine(deltaWeightHidden.Length);

            Console.WriteLine("DELTA WEIGHTS INPUT HIDDEN:");
            Console.WriteLine(deltaWeightHidden.ToString<double>());

            //DONT DELETE
            double[,] deltaWeightHidden2D = Matrix.Create(errors_hidden); //Convert to Matrix

            Console.WriteLine("DELTA WEIGHTS INPUT HIDDEN TO MATRIX:");
            Console.WriteLine(deltaWeightHidden.ToString<double>());

            //DONT DELETE
            //deltaWeightHidden2D = InputMatrix.Dot(deltaWeightHidden2D);
            deltaWeightHidden2D = deltaWeightHidden2D.Dot(InputMatrix);
            deltaWeightHidden2D = deltaWeightHidden2D.Dot(InputMatrix);
            Console.WriteLine("DELTA WEIGHTS INPUT HIDDEN TRANSPOSED:");
            Console.WriteLine(deltaWeightHidden.ToString<double>());

            NaNCheck("SGD: Calculate Delta of the weight between input and hidden"); 
            /*---------------------------------------------------------------
             * --        Adjust Weights and Biases using the delta         --
             ---------------------------------------------------------------*/
            //The Biases just get adjusted by adding the Errors multiplied by the learning rate
            BiasOutput = BiasOutput.Add(errors_output.Multiply(learningRate)); //Output Bias
            NaNCheck("SGD: Adjust Bias Output"); 


            NaNCheck("SGD: Adjust W Hidden output");
            //TODO: MAYBE CLAMP WEIGHTS
            BiasHidden = BiasHidden.Add(errors_hidden.Multiply(learningRate)); //Hidden Bias
            //ClampWeightsAndBiases();
            NaNCheck("SGD: Adjust bias hidden");

            Console.BackgroundColor = ConsoleColor.Cyan;

            Console.WriteLine("Delta Weight Hidden MATRIX LENGTH: "
                              + deltaWeightHidden.Length
                     );

            Console.WriteLine("Delta Weight Hidden MATRIX TOTAL LENGTH: "
                              + deltaWeightHidden.GetTotalLength()
                             );

            Console.WriteLine("Weights I Hidden MATRIX LENGTH: "
                              + WeightsInputHidden.Length
                             );

            Console.WriteLine("Weights I Hidden MATRIX TOTAL LENGTH: "
                              + WeightsInputHidden.GetTotalLength()
                             );

            Console.BackgroundColor = ConsoleColor.Black;
            /* IMPORTANT LINE DONT DELETE*/
            WeightsInputHidden = WeightsInputHidden.Add(deltaWeightHidden2D.Multiply(learningRate));
            /**/

            //NaNCheck("SGD: Adjust w input hidden"); 


            //ClampWeightsAndBiases();
            NaNCheck("SGD: CLAMPED"); 

           
        }

        private double GenerateXORLabel(double v1, double v2)
        {
            if ((v1 > 0.0)^(v2 > 0.0))
            {
                return 1.0;
            }
            return 0.0;
        }

        public (double[][] data, double[][] labels) generateXORData(int Amount)
        {
            List<double[]> Data = new List<double[]>();
            List<double[]> LabelsList = new List<double[]>();

            for (int i = 0; i < Amount; i++){
                
                double[][] TrainData = {
                    new double[]{0, 1},
                    new double[]{1, 1},
                    new double[]{1, 0},
                    new double[]{0, 0},
                };
                
                TrainData.Shuffle();
                
                var Labels = new double[][] { new double[] { 0 } };
                Labels[0][0] = GenerateXORLabel(TrainData[0][0], TrainData[0][1]);
                
                Data.Add(TrainData[0]);
                LabelsList.Add(Labels[0]);
            }

            return (Data.ToArray(), LabelsList.ToArray());

        }

        public void SetWeightsAndBiases(double[,] NewWeightsInput, double[,] NewWeightsOutput,
                                        double[] NewBiasHidden, double[] NewBiasOutput){
            WeightsInputHidden = NewWeightsInput;
            WeightsHiddenOutput = NewWeightsOutput;
            BiasHidden = NewBiasHidden;
            BiasOutput = NewBiasOutput;
        }

        void ClampWeightsAndBiases()
        {
            WeightsInputHidden = WeightsInputHidden.Apply((arg) => Math.Clamp(arg, -1.0, 1.0));
            WeightsHiddenOutput = WeightsHiddenOutput.Apply((arg) => Math.Clamp(arg, -1.0, 1.0));
            BiasHidden = BiasHidden.Apply((arg) => Math.Clamp(arg, -1.0, 1.0));
            BiasOutput = BiasOutput.Apply((arg) => Math.Clamp(arg, -1.0, 1.0));
            //WeightsInputHidden.ApplyInPlace((arg) => Math.Clamp(arg, -1.0, 1.0));
       
        }

        void fixWeights(){
            WeightsInputHidden = WeightsInputHidden.Apply((arg) => RemovePrecicion(arg));
            WeightsHiddenOutput = WeightsHiddenOutput.Apply((arg) => RemovePrecicion(arg));
        }

        double RemovePrecicion(double a)
        { 
            if(a > 0){
                if(a < 1E-100){
                    Console.WriteLine("!!!_-: + " + a);
                    return 1E-100;
                }
            }

            if(a > 1E50){
                Console.WriteLine("TO BIG: + " + a);
                return 1.0;
                
            }

            return a;

        
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

            //This is an ugly fix for NaN
            //if (double.IsNaN(x))
                //Relu = 0;
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

        public void NaNCheck(string extraMessage){
            if (WeightsInputHidden.HasNaN() || WeightsInputHidden.HasInfinity()){
                Console.BackgroundColor = ConsoleColor.Red;
                Console.WriteLine("WeightsInputHidden" + " has NAN! " + extraMessage);
                Console.BackgroundColor = ConsoleColor.Blue;
                Console.WriteLine(Output.ToString<double>());
                Visualize();
                System.Threading.Thread.Sleep(10000);
            }

            if (WeightsHiddenOutput.HasNaN() || WeightsHiddenOutput.HasInfinity())
            {
                Console.BackgroundColor = ConsoleColor.Red;
                Console.WriteLine("WeightsHiddenOutput" + " has NAN! " + extraMessage);
                Console.WriteLine(WeightsHiddenOutput.ToString<double>());
                Console.BackgroundColor = ConsoleColor.Blue;
                Console.WriteLine(Output.ToString<double>());
                Visualize();
                System.Threading.Thread.Sleep(10000);
            }

            if (Inputs.HasNaN() || Inputs.HasInfinity())
            {
                Console.BackgroundColor = ConsoleColor.Red;
                Console.WriteLine("Inputs" + " has NAN! " + extraMessage);
                Console.BackgroundColor = ConsoleColor.Blue;
            }

            if (Hidden.HasNaN() || Hidden.HasInfinity())
            {
                Console.BackgroundColor = ConsoleColor.Red;
                Console.WriteLine("Hidden" + " has NAN! " + extraMessage);
                Console.BackgroundColor = ConsoleColor.Blue;
                System.Threading.Thread.Sleep(10000);
            }

            if (HiddenWithoutRElu.HasNaN() || HiddenWithoutRElu.HasInfinity())
            {
                Console.BackgroundColor = ConsoleColor.Red;
                Console.WriteLine("HiddenWithoutRElu" + " has NAN! " + extraMessage);
                Console.BackgroundColor = ConsoleColor.Blue;
            }

            if (BiasHidden.HasNaN() || BiasHidden.HasInfinity())
            {
                Console.BackgroundColor = ConsoleColor.Red;
                Console.WriteLine("BiasHidden" + " has NAN! " + extraMessage);
                Console.BackgroundColor = ConsoleColor.Blue;
            }

            if (BiasOutput.HasNaN() || BiasOutput.HasInfinity())
            {
                Console.BackgroundColor = ConsoleColor.Red;
                Console.WriteLine("BiasOutput" + " has NAN! " + extraMessage);
                Console.BackgroundColor = ConsoleColor.Blue;
            }
            if (Output.HasNaN() || Output.HasInfinity())
            {
                Console.BackgroundColor = ConsoleColor.Red;
                Console.WriteLine("Output" + " has NAN! " + extraMessage);
                Console.WriteLine(Output.ToString<double>());
                Console.BackgroundColor = ConsoleColor.Blue;
                System.Threading.Thread.Sleep(10000);
            }
        }

        public void Visualize()
        {
            Console.WriteLine("--- - - - NEURAL NETWORK - - - ---");
            Console.WriteLine("Inputs:");
            Console.WriteLine(Inputs.ToString<double>());


            Console.WriteLine("Weights Input Hidden:");
            Console.BackgroundColor = ConsoleColor.Yellow;
            var MaxValue1 = WeightsInputHidden.ToString<double>();
            var MinValue1 = WeightsHiddenOutput.ToString<double>();
            Console.WriteLine(MaxValue1);
            Console.BackgroundColor = ConsoleColor.Black;
            Console.WriteLine("Weights Hidden Output:");
            Console.WriteLine(MinValue1);

            Console.WriteLine("Hidden (ReLU):");
            Console.WriteLine(Hidden.ToString<double>());

            Console.WriteLine("Hidden (Without ReLU):");
            Console.WriteLine(HiddenWithoutRElu.ToString<double>());

            Console.WriteLine("WeightsOutput Max, Min:");
            var MaxValue = WeightsHiddenOutput.Max();
            var MinValue = WeightsHiddenOutput.Min();
            Console.WriteLine(MaxValue);
            Console.WriteLine(MinValue);

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
