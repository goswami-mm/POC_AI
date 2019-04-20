package com.manmohan.pocai.aiengine.env.tracking;

import com.manmohan.pocai.aiengine.env.Box;

import java.util.ArrayList;
import java.util.List;

public class MovementTracker {
    private String name;
    private List<Box> moves = new ArrayList<>();

    public MovementTracker(String name){
        this.name = name;
    }

    public void addMove(Box box){
        moves.add(box);
    }

    public void clearMoves(){
        moves.clear();
    }

    public List<Box> getMoves(){
        return moves;
    }

    public String getName(){
        return name;
    }
}
