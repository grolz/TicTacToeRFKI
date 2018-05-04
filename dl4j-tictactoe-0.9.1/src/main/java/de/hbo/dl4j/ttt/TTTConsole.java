
package de.hbo.dl4j.ttt;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

import de.hbo.dl4j.ttt.classifier.AbstractTTTClassifierPlayer;
import de.hbo.tictactoe.TTTGame;
import de.hbo.tictactoe.TTTGame.Prompter;
import de.hbo.tictactoe.TTTGame.TTTPlayer;
import de.hbo.tictactoe.TicTacToe;

public class TTTConsole {

    public static void main(String[] args) throws Exception {
    
    	TTTGame.Prompter prompter = new ConsolePrompter();
    	
		boolean done = false;
    	while(!done) {
    		
    		// type of players
    		TTTGame.TTTPlayer[] players = null;
    		while (players == null) try {
	    		
    			String input = prompter.promptInput("enter number of players [1]: ");
	        	int numberOfPlayers = input.length()==0? 1 : Integer.parseInt(input);
	        	if (numberOfPlayers < 0 || numberOfPlayers > 2) throw new Exception("number of players must be between 0 and 2.");
	        	
	        	players = new TTTGame.TTTPlayer[2];
	        	players[0] = numberOfPlayers>0? new TTTConsolePlayer() : new TTTComputerPlayer();
	        	players[1] = numberOfPlayers<2? new TTTComputerPlayer() : new TTTConsolePlayer();
	        
    		} catch (Exception e) {System.out.println(e);}
    		
    		
    		// starting player
    		int startingPlayer = -1;
    		while (startingPlayer <0) try {
	    		
    			String input = prompter.promptInput("enter player to start [0]: ");
	        	int index = input.length()==0? 0 : Integer.parseInt(input);
	        	if (index < 0 || index > 1) throw new Exception("player to start is either 0 or 1.");
	        	
	        	startingPlayer = index; 
	        	
    		} catch (Exception e) {System.out.println(e);}

    		
    		// actual game
    		TTTGame.play(players, startingPlayer, prompter, System.out);


        	// restart or quit
        	String answer = null;
    		while (answer == null) try {
    			
    			String input = prompter.promptInput("play another one (y/n) [y]: ");
	        	if (input.length()==0) input = "y";
	        	if (!input.equals("y") && !input.equals("n")) throw new Exception("answer must be 'y' or 'n'.");
	        	
	        	done = (answer = input).equals("n"); 
	        	
    		} catch (Exception e) {System.out.println(e);}
    	
    	}
    	
        System.out.println("... done.");
    }
    
    private static class TTTConsolePlayer implements TTTPlayer {

		@Override
		public void draw(TicTacToe ttt, Prompter prompter) {
			
			boolean done = false;
			
    		while (!done) {
    			
    			String input = prompter.promptInput("move (1-9): ");
	        	int move = Integer.parseInt(input)-1;
	        	for (int i = 0; i < ttt.getAvailableMoves().length; i++) if (move==ttt.getAvailableMoves()[i]) {
	        		ttt.move(move);
	        		done = true;
	        	} 
	        	
    		}
		}
    }
    
    private static class TTTComputerPlayer extends AbstractTTTClassifierPlayer.Model1ComputerPlayer {};
    
    private static class ConsolePrompter implements TTTGame.Prompter {
    	
    	private final static BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
    	
        @Override
		public String promptInput(String prompt) {
        	
        	System.out.print(prompt);
        	
    		try {
				return reader.readLine();
			} catch (IOException e) {
				throw new RuntimeException(e);
			}
        }	
    }
}




























