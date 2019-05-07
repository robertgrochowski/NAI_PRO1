package nai.pro1;

import Jama.Matrix;

import java.util.ArrayList;
import java.util.List;

class NeuralNetwork {

    private Matrix weights;
    private double alpha;
    private Matrix bias;

    private static final Matrix trainingSet = new Matrix(new double[][]{
            {1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1}, //2
            {0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0}, //0
            {1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1}, //2
            {0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0}, //1
            {0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0}, //1
            {1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1}, //0
            {0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0}, //1
            {0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0}, //1
            {0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0}  //2
    });

    private static final Matrix answerSet = new Matrix(new double[][]{
            {0, 0, 1},
            {1, 0, 0},
            {0, 0, 1},
            {0, 1, 0},
            {0, 1, 0},
            {1, 0, 0},
            {0, 1, 0},
            {0, 1, 0},
            {0, 0, 1}
    });

    NeuralNetwork(int neuronAmount, double alpha)
    {
        this.weights = Matrix.random(neuronAmount, 24);
        this.alpha = alpha;

        bias = Matrix.random(neuronAmount, 1);
    }

    //Returns epoch error list values to View controller
    List<Double> teach(int epochAmount, double maxError) {

        List<Double> errorList = new ArrayList<>();
        double error = 100;
        int epochNumber = 0;

        while(error >= maxError && epochNumber++ <= epochAmount) {

            //Choose X
            for(int s = 0; s < trainingSet.getRowDimension(); s++)
            {
                //thisInput - Xn
                Matrix thisInput = trainingSet.getMatrix(s, s, 0, trainingSet.getColumnDimension() - 1).transpose();
                //thisAnswer - Dn
                Matrix thisAnswer = answerSet.getMatrix(s, s, 0, answerSet.getColumnDimension() - 1).transpose();
                //NETValue - All neurons
                Matrix NETValue = weights.times(thisInput).plus(bias);

                //Output for all neurons
                Matrix output = getFunctionValue(NETValue);

                //compute error
                error = getError(output, thisAnswer);

                //improve
                weights = thisAnswer.minus(output).
                        times(thisInput.transpose()).
                        times(alpha).
                        plus(weights);

                bias = thisAnswer.minus(output).times(alpha).plus(bias);
            }

            errorList.add(error);
        }
        return errorList;
    }

    //Classify class depending on output
    int classify(Matrix input)
    {
        Matrix NETValue = weights.times(input.transpose()).plus(bias);
        Matrix output = getFunctionValue(NETValue);
        output.print(1,0);

        int sum = 0;
        for(int i = 0; i < output.getRowDimension(); i++)
            sum += output.get(i, 0);

        if(sum != 1)
            return -1;

        for(int i = 0; i < output.getRowDimension(); i++)
            if(output.get(i, 0) == 1)
                return i;
        return -1;
    }

    //Compute neural network error
    private double getError(Matrix outputs, Matrix answerSet) {
        double error = 0;
        Matrix diff = answerSet.minus(outputs);

        for (int row = 0; row < diff.getRowDimension(); row++)
            for(int col = 0; col < diff.getColumnDimension(); col++)
                error += Math.pow(diff.get(row, col), 2);

        return error * 0.5;
    }
    //Compute discrete unipolar Y (output) value
    private Matrix getFunctionValue(Matrix NETValue) {
        Matrix output = new Matrix(NETValue.getRowDimension(), 1);
        for(int i = 0; i < NETValue.getRowDimension(); i++)
        {
            if(NETValue.get(i, 0 ) >= 0)
                output.set(i, 0, 1d);
            else output.set(i, 0, 0d);
        }
        return output;
   }
}
