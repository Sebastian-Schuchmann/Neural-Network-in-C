using System;
using System.Collections.Generic;

namespace NeuralNetwork{
    
    class Perceptron
    {
        float[] Inputs;
        float[] Weights;
        
        //Hyperparameters
        float learningRate;
        
        Random rndGenerator;
        
        public Perceptron(int Size, float lr)
        {
            //+1 for Bias Weight
            Inputs = new float[Size + 1];
            Weights = new float[Size + 1];
            
            learningRate = lr;
            
            rndGenerator = new Random();
            InitializeWeights();
        }
        
        public void SetInputs(List<float> inpts)
        {
            //Adding Bias
            inpts.Add(1f);
            Inputs = inpts.ToArray();
        }
        
        public float Calculate()
        {
            float sum = 0f;
            for (int i = 0; i < Inputs.Length; i++)
            {
                sum += Inputs[i] * Weights[i];
            }
            return ActivationFunction(sum);
        }
        
        public float CalculateError(float Desired, float ActualOutput)
        {
            return Desired - ActualOutput;
        }
        
        public void AdjustWeights(float Error)
        {
            for (int i = 0; i < Weights.Length; i++)
                Weights[i] = Weights[i] + (Error * Inputs[i] * learningRate);
        }
        
        float GetBias()
        {
            return Inputs[Inputs.Length - 1] * Weights[Weights.Length - 1];
        }
        
        public float CalculateInput(List<float> inpts)
        {
            SetInputs(inpts);
            return Calculate();
        }
        
        void InitializeWeights()
        {
            for (int i = 0; i < Weights.Length; i++)
                Weights[i] = (float)rndGenerator.NextDouble() * 2.0f - 1.0f;
        }
        
        float ActivationFunction(float val)
        {
            val = MathF.Sign(val);
            if (val == 0.0f)
                return 1.0f;
            return val;
        }
    }
}











