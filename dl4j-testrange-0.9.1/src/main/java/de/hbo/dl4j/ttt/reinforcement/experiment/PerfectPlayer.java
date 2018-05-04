package de.hbo.dl4j.ttt.reinforcement.experiment;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import de.hbo.tictactoe.TTTGame.Prompter;
import de.hbo.tictactoe.TTTGame.TTTPlayer;
import de.hbo.tictactoe.TicTacToe;

public class PerfectPlayer implements TTTPlayer {
	
	private final Map<String, int[]> table;
	
	public PerfectPlayer() throws Exception {
		
		table = new HashMap<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(new File("src/main/resources/classification/tictactoe.txt")))) {
        	while (reader.ready()) {
        		String line = reader.readLine();
        		if (line == null || line.length() == 0) break;
        		
        		String key = line.substring(0, line.indexOf('|'));
        		String value = line.substring(line.indexOf('|')+1);
        		List<Integer> values = new ArrayList<>();
        		for (int i = 0; i < value.length(); i++) if (value.charAt(i)=='1') values.add(i);
        		int[] v = new int[values.size()];
        		for (int i = 0; i < values.size(); i++) v[i] = values.get(i).intValue();
        		
        		table.put("[" + key.substring(0, key.lastIndexOf(',')) + "]", v);
        	}
        }
	}
	
	@Override public void draw(TicTacToe ttt, Prompter prompter) {this.draw(ttt);}
	
	private void draw(TicTacToe ttt) {
		int[] moves = table.get(TTTUtil.toString(ttt.getBoard()));
		ttt.move(moves[new Random(System.currentTimeMillis()).nextInt(moves.length)]);
	}
}