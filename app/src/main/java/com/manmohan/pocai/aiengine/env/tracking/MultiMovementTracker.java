package com.manmohan.pocai.aiengine.env.tracking;

import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;

import com.manmohan.pocai.aiengine.Classifier;
import com.manmohan.pocai.aiengine.env.Box;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static com.manmohan.pocai.aiengine.env.tracking.MultiBoxTracker.COLORS;

public class MultiMovementTracker {
    private static final float BOX_SIZE = 50;
    private Map<String, List<Box>> trackerMap = new HashMap<>();

    public void addNewResult(final List<Classifier.Recognition> results){
        for(Classifier.Recognition recognition : results){
            if(trackerMap.containsKey(recognition.getTitle())){
                List<Box> arrayList = trackerMap.get(recognition.getTitle());
                Box box = new Box();
                box.x = recognition.getLocation().centerX();
                box.y = recognition.getLocation().centerY();
                if(arrayList == null)
                    arrayList = new ArrayList<>();

                arrayList.add(box);
                trackerMap.put(recognition.getTitle(), arrayList);
            } else {
                List<Box> arrayList = new ArrayList<>();
                Box box = new Box();
                box.x = recognition.getLocation().centerX();
                box.y = recognition.getLocation().centerY();

                arrayList.add(box);
                trackerMap.put(recognition.getTitle(), arrayList);
            }
        }
    }

    public void clearAll(){
        trackerMap.clear();
    }

    public void draw(Canvas canvas){
        int i = 0;
        for(Map.Entry<String, List<Box>> entry : trackerMap.entrySet()){
            int color =  COLORS[i];
            Paint paint = new Paint();
            paint.setColor(color);
            for(Box b: entry.getValue()) {
                RectF rectF = new RectF();
                rectF.left = b.x;
                rectF.top = b.y;
                rectF.bottom = b.y + BOX_SIZE;
                rectF.right = b.x + BOX_SIZE;
                canvas.drawRoundRect(rectF, 1, 1, paint);
            }
            i++;
        }
    }
}
