package de.hbo.dl4j.ttt.reinforcement.experiment;

import java.util.ArrayList;
import java.util.List;

import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.learning.ILearning.LConfiguration;
import org.deeplearning4j.rl4j.learning.sync.ExpReplay;
import org.deeplearning4j.rl4j.learning.sync.IExpReplay;
import org.deeplearning4j.rl4j.learning.sync.Transition;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.policy.DQNPolicy;
import org.deeplearning4j.rl4j.policy.EpsGreedy;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.util.Constants;
import org.deeplearning4j.rl4j.util.DataManager;
import org.deeplearning4j.rl4j.util.DataManager.StatEntry;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.Value;

public class TTTQLearning<O extends Encodable> {
    
    //**********************************************************************************************************************************************

    @Getter private int stepCounter = 0;
    @Getter final private QLConfiguration configuration;
    @Getter final private MDP<O, Integer, DiscreteSpace> mdp;
    @Getter private DQNPolicy<O> policy;
    
    final private IExpReplay<Integer> expReplay;
    final private IDQN<?> currentDQN;
    
    private int epochCounter = 0;
    private IHistoryProcessor historyProcessor = null;
    private int lastSave = -Constants.MODEL_SAVE_FREQ;	
    private EpsGreedy<O, Integer, DiscreteSpace> egPolicy;
    private IDQN<?> targetDQN;
    private int lastAction;
    private INDArray history[] = null;
    private double accuReward = 0;
    private int lastMonitor = -Constants.MONITOR_FREQ;

    /*
    public TTTQLearning(MDP<O, Integer, DiscreteSpace> mdp, IDQN<?> dqn, QLConfiguration conf, DataManager dataManager) {this(mdp, dqn, conf, dataManager, conf.getEpsilonNbStep());}
    public TTTQLearning(MDP<O, Integer, DiscreteSpace> mdp, DQNFactory factory, QLConfiguration conf, DataManager dataManager) {this(mdp, factory.buildDQN(mdp.getObservationSpace().getShape(), mdp.getActionSpace().getSize()), conf, dataManager);}
    public TTTQLearning(MDP<O, Integer, DiscreteSpace> mdp, DQNFactoryStdDense.Configuration netConf, QLConfiguration conf, DataManager dataManager) {this(mdp, new DQNFactoryStdDense(netConf), conf, dataManager);}
    */
    
    public TTTQLearning(
		MDP<O, Integer, DiscreteSpace> mdp, 
		IDQN<?> dqn, 
		QLConfiguration conf, 
		int epsilonNbStep
	) {
        expReplay = new ExpReplay<>(conf.getExpRepMaxSize(), conf.getBatchSize(), conf.getSeed());
        this.configuration = conf;
        this.mdp = mdp;
        currentDQN = dqn;
        targetDQN = dqn.clone();
        policy = new DQNPolicy<>(currentDQN);
        //egPolicy = new EpsGreedy<>(policy, mdp, conf.getUpdateStart(), epsilonNbStep, new Random(conf.getSeed()), conf.getMinEpsilon(), this);
        mdp.getActionSpace().setSeed(conf.getSeed());
    }
    
    //**********************************************************************************************************************************************

    public void train() {

        try {

            while (stepCounter < getConfiguration().getMaxStep()) {
            	

            	// *********************************************************************************** preepoch 
                history = null;
                lastAction = 0;
                accuReward = 0;
            	
                // *********************************************************************************** epoch 
                
                
                
                currentDQN.reset(); // tut hier nix, ist anscheinend nur für recurrent nets
                
                 
                
                O obs;
                double reward;
                int step;
                // *********************************************************************************** initMdp
                {
                	IHistoryProcessor hp = historyProcessor;

                	O o = mdp.reset(); // Initialisierung des MDP
                    O nextO = o;

                    int s = 0;
                    double r = 0;

                    int skipFrame = hp!=null ? hp.getConf().getSkipFrame() : 1;
                    int requiredFrame = hp!=null ? skipFrame * (hp.getConf().getHistoryLength() - 1) : 0;

                    while (s < requiredFrame) {
                    	
                    	INDArray input;
                    	{
                            INDArray arr = Nd4j.create(o.toArray());
                            int[] shape = mdp.getObservationSpace().getShape();
                            input = shape.length == 1? arr:arr.reshape(shape);
                    	}

                        if (hp!=null) hp.record(input);

                        Integer action = mdp.getActionSpace().noOp(); //by convention should be the NO_OP
                        if (s % skipFrame == 0 && hp!=null) hp.add(input);

                        StepReply<O> stepReply = mdp.step(action);
                        r += stepReply.getReward();
                        nextO = stepReply.getObservation();

                        s++;
                    }

                    obs = nextO;
                    reward = r;
                    step = s;
                }
                // *********************************************************************************** initMdp


                Double startQ = Double.NaN;
                double meanQ = 0;
                int numQ = 0;
                
                List<Double> scores = new ArrayList<>();
                while (/*step < getConfiguration().getMaxEpochStep() &&*/ !mdp.isDone()) {

                    if (stepCounter % configuration.getTargetDqnUpdateFreq() == 0) targetDQN=currentDQN.clone();

                    // *********************************************************************************** trainStep
                    QLStepReturn<O> stepR; 
                    {

                        Integer action;
                        INDArray input = getInput(mdp.getObservationSpace().getShape(), obs); // reshapes obs if necessary
                        boolean isHistoryProcessor = historyProcessor != null;


                        if (isHistoryProcessor) historyProcessor.record(input);

                        int skipFrame = isHistoryProcessor ? historyProcessor.getConf().getSkipFrame() : 1;
                        int historyLength = isHistoryProcessor ? historyProcessor.getConf().getHistoryLength() : 1;
                        int updateStart = configuration.getUpdateStart() + ((configuration.getBatchSize() + historyLength) * skipFrame);

                        Double maxQ = Double.NaN; //ignore if Nan for stats

                        
                        //if step of training, just repeat lastAction
                        if (stepCounter % skipFrame != 0) action = lastAction;
                        else {
                            
                        	if (history == null) {
                                if (isHistoryProcessor) {
                                    historyProcessor.add(input);
                                    history = historyProcessor.getHistory();
                                } else history = new INDArray[] {input};
                            }
                            
                            //concat the history into a single INDArray input
                            INDArray hstack = Transition.concat(Transition.dup(history));

                            //if input is not 2d, you have to append that the batch is 1 length high
                            if (hstack.shape().length > 2) {

                                int[] nshape = new int[hstack.shape().length + 1];
                                nshape[0] = 1;
                                for (int i = 0; i < hstack.shape().length; i++) {
                                    nshape[i + 1] = hstack.shape()[i];
                                }
                            	
                            	hstack = hstack.reshape(nshape);
                            }

                            INDArray qs = currentDQN.output(hstack); //------------------------------------------------------------------------------------------------------------------ OUTPUT !!!
                            
                            int maxAction = Nd4j.argMax(qs, Integer.MAX_VALUE).getInt(0);

                            maxQ = qs.getDouble(maxAction);
                            action = egPolicy.nextAction(hstack);  //-------------------------------------------------------------------------------------------------------------------- ACTION SELECTION !!!
                        }

                        lastAction = action;

                        StepReply<O> stepReply = mdp.step(action);  //------------------------------------------------------------------------------------------------------------------- STEP !!!

                        accuReward += stepReply.getReward() * configuration.getRewardFactor();

                        //if it's not a skipped frame, you can do a step of training
                        if (stepCounter % skipFrame == 0 || stepReply.isDone()) {

                            INDArray ninput = getInput(mdp.getObservationSpace().getShape(), stepReply.getObservation());

                            if (isHistoryProcessor) historyProcessor.add(ninput);

                            INDArray[] nhistory = isHistoryProcessor ? historyProcessor.getHistory() : new INDArray[] {ninput};

                            expReplay.store(new Transition<>(history, action, accuReward, stepReply.isDone(), nhistory[0]));

                            if (stepCounter > updateStart) {
                                
                                {//setTarget() 
                                	
                                	ArrayList<Transition<Integer>> transitions = expReplay.getBatch();

                                    int[] nshape = makeShape(transitions.size(), historyProcessor == null ? mdp.getObservationSpace().getShape() : historyProcessor.getConf().getShape());
                                    
                                    INDArray nextObs = Nd4j.create(nshape);
                                    int[] actions = new int[transitions.size()];
                                    boolean[] areTerminal = new boolean[transitions.size()];

                                    
                                    // Input für "fit" erzeugen
                                    INDArray o = Nd4j.create(nshape);
                                    for (int i = 0; i < transitions.size(); i++) {
                                    	
                                        Transition<Integer> t = transitions.get(i);
                                        
                                        areTerminal[i] = t.isTerminal();
                                        actions[i] = t.getAction();
                                        
                                        INDArray[] obsArray = t.getObservation();
                                        if (o.rank() == 2) o.putRow(i, obsArray[0]);
                                        else for (int j = 0; j < obsArray.length; j++) o.put(new INDArrayIndex[] {NDArrayIndex.point(i), NDArrayIndex.point(j)}, obsArray[j]);

                                        INDArray[] nextObsArray = Transition.append(t.getObservation(), t.getNextObservation());
                                        if (nextObs.rank() == 2) nextObs.putRow(i, nextObsArray[0]);
                                        else for (int j = 0; j < nextObsArray.length; j++) nextObs.put(new INDArrayIndex[] {NDArrayIndex.point(i), NDArrayIndex.point(j)}, nextObsArray[j]);
                                    }

                                    
                                    // Labels für "fit" erzeugen
                                    INDArray dqnOutputNext = currentDQN.output(nextObs);
                                    INDArray dqnOutputAr = currentDQN.output(o);
                                    
                                    if (configuration.isDoubleDQN()) {

                                        INDArray targetDqnOutputNext = targetDQN.output(nextObs);
                                        INDArray getMaxAction = Nd4j.argMax(dqnOutputNext, 1);
                                        
                                        for (int i = 0; i < transitions.size(); i++) {
                                            double yTar = transitions.get(i).getReward() + (!areTerminal[i]? getConfiguration().getGamma() * targetDqnOutputNext.getDouble(i, getMaxAction.getInt(i)) : 0d);
                                            double previousV = dqnOutputAr.getDouble(i, actions[i]);
                                            dqnOutputAr.putScalar(i, actions[i], Math.min(previousV + configuration.getErrorClamp(), Math.max(yTar, previousV - configuration.getErrorClamp())));
                                        }
                                        
                                    } else {
                                    	
                                    	INDArray tempQ = Nd4j.max(dqnOutputNext, 1);                                    
                                        
                                    	for (int i = 0; i < transitions.size(); i++) {
                                            double yTar = transitions.get(i).getReward() + (!areTerminal[i]? getConfiguration().getGamma() * tempQ.getDouble(i) : 0d);
                                            double previousV = dqnOutputAr.getDouble(i, actions[i]);
                                            dqnOutputAr.putScalar(i, actions[i], Math.min(previousV + configuration.getErrorClamp(), Math.max(yTar, previousV - configuration.getErrorClamp())));
                                        }
                                    }

                                    currentDQN.fit(o, dqnOutputAr); //---------------------------------------------------------------------------------------------------------------- FIT !!!
                                }
                                
                                
                            }

                            history = nhistory;
                            accuReward = 0;
                        }


                        stepR = new QLStepReturn<>(maxQ, currentDQN.getLatestScore(), stepReply);

                    }
                    // *********************************************************************************** trainStep
                    

                    if (!stepR.getMaxQ().isNaN()) {
                        if (startQ.isNaN()) startQ = stepR.getMaxQ();
                        numQ++;
                        meanQ += stepR.getMaxQ();
                    }

                    if (stepR.getScore() != 0) scores.add(stepR.getScore());

                    reward += stepR.getStepReply().getReward();
                    obs = stepR.getStepReply().getObservation();

                    stepCounter++;
                    step++;
                }

                meanQ /= (numQ + 0.001); //avoid div zero


                DataManager.StatEntry statEntry = new QLStatEntry(stepCounter, epochCounter, reward, step, scores, egPolicy.getEpsilon(), startQ, meanQ);
        		
                // *********************************************************************************** postepoch

                if (historyProcessor != null) historyProcessor.stopMonitor();
            	
                // *********************************************************************************** epoch 

                epochCounter++;

                if (stepCounter - lastSave >= Constants.MODEL_SAVE_FREQ) {
                    //FIXME: dataManager.save((Learning)this);
                    lastSave = stepCounter;
                }
            }
            
        } catch (Exception e) {e.printStackTrace();}
    }

    public static <O extends Encodable> INDArray getInput(int[] shape, O obs) {
        INDArray arr = Nd4j.create(obs.toArray());
        if (shape.length == 1) return arr;
        return arr.reshape(shape);
    }

    public static int[] makeShape(int size, int[] shape) {
        int[] nshape = new int[shape.length + 1];
        nshape[0] = size;
        for (int i = 0; i < shape.length; i++) nshape[i + 1] = shape[i];
        return nshape;
    }
    
    //**********************************************************************************************************************************************

    @AllArgsConstructor
    @Builder
    @Value
    public static class QLStatEntry implements StatEntry {
        int stepCounter;
        int epochCounter;
        double reward;
        int episodeLength;
        List<Double> scores;
        float epsilon;
        double startQ;
        double meanQ;
    }

    @AllArgsConstructor
    @Builder
    @Value
    public static class QLStepReturn<O> {
        Double maxQ;
        double score;
        StepReply<O> stepReply;
    }

    @Data
    @AllArgsConstructor
    @Builder
    @EqualsAndHashCode(callSuper = false)
    public static class QLConfiguration implements LConfiguration {
        int seed;
        int maxEpochStep;
        int maxStep;
        int expRepMaxSize;
        int batchSize;
        int targetDqnUpdateFreq;
        int updateStart;
        double rewardFactor;
        double gamma;
        double errorClamp;
        float minEpsilon;
        int epsilonNbStep;
        boolean doubleDQN;
    }
}