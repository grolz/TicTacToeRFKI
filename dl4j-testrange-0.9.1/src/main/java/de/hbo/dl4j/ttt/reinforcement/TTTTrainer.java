package de.hbo.dl4j.ttt.reinforcement;

import java.io.IOException;

import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdDense;
import org.deeplearning4j.rl4j.util.DataManager;


public class TTTTrainer {

    public static void main(String[] args) throws IOException {

        new TTTQLearningDiscreteDense<>(
    		new TTTMDP(), 
    		DQNFactoryStdDense.Configuration.builder().l2(0.01).learningRate(1e-2).numLayer(3).numHiddenNodes(16).build(), 
    		new TTTQLearningDiscreteDense.QLConfiguration(
		        123,   	//seed: 				Random seed
		        9,		//maxEpochStep: 		Max step By epoch
		        100, 	//maxStep:				Max step
		        10, 	//expRepMaxSize:		Max size of experience replay
		        1,    	//batchSize:			size of batches
		        100,   	//targetDqnUpdateFreq:	target update (hard)
		        0,     	//updateStart:			num step noop warmup
		        0.05,  	//rewardFactor:			reward scaling
		        0.99,  	//gamma:				gamma
		        10.0,  	//errorClamp:			td-error clipping
		        0.1f,  	//minEpsilon:			min epsilon
		        2000,  	//epsilonNbStep:		num step for eps greedy anneal
		        true   	//doubleDQN:			double DQN
		    ),
    		new DataManager()
		).train();
    }
}