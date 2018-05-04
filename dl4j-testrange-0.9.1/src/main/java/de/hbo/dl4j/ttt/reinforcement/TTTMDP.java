package de.hbo.dl4j.ttt.reinforcement;

import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.space.ObservationSpace;
import org.json.JSONObject;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import de.hbo.dl4j.ttt.reinforcement.TTTMDP.TTTState;
import de.hbo.tictactoe.TicTacToe;
import lombok.Value;
import lombok.extern.slf4j.Slf4j;

@Slf4j
public class TTTMDP implements MDP<TTTState, Integer, DiscreteSpace> {
    
    
    private DiscreteSpace actionSpace = new DiscreteSpace(9);
	private final TTTObservationSpace observationSpace = new TTTObservationSpace();
    private TicTacToe ttt;

    
    public TTTMDP() {}
    

    @Override public TTTMDP newInstance() {return new TTTMDP();}
    
    @Override public TTTState reset() {
    	this.ttt = new TicTacToe();
    	return this.getCurrentState();
	}
    
    @Override
	public StepReply<TTTState> step(Integer a) {
    	int current = this.ttt.getNext(); 
		if (this.ttt.move(a)) return new StepReply<>(this.getCurrentState(), this.ttt.isWin()*current, ttt.isFinal(), new JSONObject("{}"));
		return new StepReply<>(this.getCurrentState(), -10, true, new JSONObject("{}"));
    }		
    
    @Override public TTTObservationSpace getObservationSpace() {
    	return observationSpace;
	}


	@Override public DiscreteSpace getActionSpace() {
		return actionSpace;
	}


    @Override public void close() {}
    @Override public boolean isDone() {return ttt.isFinal();}
    
    private TTTState getCurrentState() {
    	int[] state = new int[ttt.getBoard().length+1];
    	for (int i = 0; i < this.ttt.getBoard().length; i++) state[i] = this.ttt.getBoard()[i]*this.ttt.getNext();
    	state[this.ttt.getBoard().length] = 1;
    	return new TTTState(state);
    }


	@Value
	public static class TTTState implements Encodable {
	
	    int[] state;
	
	    @Override
	    public double[] toArray() {
	    	double[] erg = new double[this.state.length];
	    	for (int i = 0; i < this.state.length; i++) erg[i] = (double)this.state[i];
	        return erg;
	    }
	}
	
	@Value
	public static class TTTObservationSpace implements ObservationSpace<TTTState> {
	
		final String name;
	    final int[] shape;
	    final INDArray low;
	    final INDArray high;
	
	    public TTTObservationSpace() {
	    	name = "TTT";
	        this.shape = new int[] {10};
	        low = Nd4j.create(10);
	        high = Nd4j.create(10);
	        for (int i = 0; i < 10; i++ ) {
	        	low.putScalar(i, -1);
	        	high.putScalar(i,  1);
	        }
	    }
	}
}
