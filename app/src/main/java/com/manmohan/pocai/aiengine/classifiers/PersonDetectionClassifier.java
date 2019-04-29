package com.manmohan.pocai.aiengine.classifiers;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.os.Trace;

import com.manmohan.pocai.aiengine.Classifier;
import com.manmohan.pocai.aiengine.env.Logger;

import org.tensorflow.Graph;
import org.tensorflow.Operation;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Vector;

public class PersonDetectionClassifier implements Classifier {
    private static final Logger LOGGER = new Logger();

    // Only return this many results.
    private static final int MAX_RESULTS = 100;
    private static final int CLASSIFIER_INPUT_SIZE = 300;
    private static final String CLASSIFIER_MODEL_FILE =
            "file:///android_asset/ssd_mobilenet_v1_android_export.pb";
    private static final String TF_OD_FACE_API_MODEL_FILE =
            "file:///android_asset/frozen_inference_graph_face.pb";
    private static final String CLASSIFIER_LABELS_FILE = "file:///android_asset/coco_labels_list.txt";

    // Config values.
    private String inputName;
    private int inputSize;

    // Pre-allocated buffers.
    private Vector<String> labels = new Vector<String>();
    private int[] intValues;
    private byte[] byteValues;
    private float[] outputLocations;
    private float[] outputScores;
    private float[] outputClasses;
    private float[] outputNumDetections;
    private String[] outputNames;

    private boolean logStats = false;

    private TensorFlowInferenceInterface inferenceInterface;

    /**
     * Initializes a native TensorFlow session for classifying images.
     *
     * @param assetManager The asset manager to be used to load assets.
     */
    public static Classifier create(
            final AssetManager assetManager) throws IOException {
        final PersonDetectionClassifier d = new PersonDetectionClassifier();

        InputStream labelsInput = null;
        String actualFilename = CLASSIFIER_LABELS_FILE.split("file:///android_asset/")[1];
        labelsInput = assetManager.open(actualFilename);
        BufferedReader br = null;
        br = new BufferedReader(new InputStreamReader(labelsInput));
        String line;
        while ((line = br.readLine()) != null) {
            LOGGER.w(line);
            d.labels.add(line);
        }
        br.close();


        d.inferenceInterface = new TensorFlowInferenceInterface(assetManager, CLASSIFIER_MODEL_FILE);

        final Graph g = d.inferenceInterface.graph();

        d.inputName = "image_tensor";
        // The inputName node has a shape of [N, H, W, C], where
        // N is the batch size
        // H = W are the height and width
        // C is the number of channels (3 for our purposes - RGB)
        final Operation inputOp = g.operation(d.inputName);
        if (inputOp == null) {
            throw new RuntimeException("Failed to find input Node '" + d.inputName + "'");
        }
        d.inputSize = CLASSIFIER_INPUT_SIZE;
        // The outputScoresName node has a shape of [N, NumLocations], where N
        // is the batch size.
        final Operation outputOp1 = g.operation("detection_scores");
        if (outputOp1 == null) {
            throw new RuntimeException("Failed to find output Node 'detection_scores'");
        }
        final Operation outputOp2 = g.operation("detection_boxes");
        if (outputOp2 == null) {
            throw new RuntimeException("Failed to find output Node 'detection_boxes'");
        }
        final Operation outputOp3 = g.operation("detection_classes");
        if (outputOp3 == null) {
            throw new RuntimeException("Failed to find output Node 'detection_classes'");
        }

        // Pre-allocate buffers.
        d.outputNames = new String[] {"detection_boxes", "detection_scores",
                "detection_classes", "num_detections"};
        d.intValues = new int[d.inputSize * d.inputSize];
        d.byteValues = new byte[d.inputSize * d.inputSize * 3];
        d.outputScores = new float[MAX_RESULTS];
        d.outputLocations = new float[MAX_RESULTS * 4];
        d.outputClasses = new float[MAX_RESULTS];
        d.outputNumDetections = new float[1];
        return d;
    }

    private PersonDetectionClassifier() {}

    @Override
    public List<Recognition> recognizeImage(final Bitmap bitmap, Matrix matrix) {
        // Log this method so that it can be analyzed with systrace.
        Trace.beginSection("recognizeImage");

        Trace.beginSection("preprocessBitmap");
        // Preprocess the image data to extract R, G and B bytes from int of form 0x00RRGGBB
        // on the provided parameters.
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        for (int i = 0; i < intValues.length; ++i) {
            byteValues[i * 3 + 2] = (byte) (intValues[i] & 0xFF);
            byteValues[i * 3 + 1] = (byte) ((intValues[i] >> 8) & 0xFF);
            byteValues[i * 3 + 0] = (byte) ((intValues[i] >> 16) & 0xFF);
        }
        Trace.endSection(); // preprocessBitmap

        // Copy the input data into TensorFlow.
        Trace.beginSection("feed");
        inferenceInterface.feed(inputName, byteValues, 1, inputSize, inputSize, 3);
        Trace.endSection();

        // Run the inference call.
        Trace.beginSection("run");
        inferenceInterface.run(outputNames, logStats);
        Trace.endSection();

        // Copy the output Tensor back into the output array.
        Trace.beginSection("fetch");
        outputLocations = new float[MAX_RESULTS * 4];
        outputScores = new float[MAX_RESULTS];
        outputClasses = new float[MAX_RESULTS];
        outputNumDetections = new float[1];
        inferenceInterface.fetch(outputNames[0], outputLocations);
        inferenceInterface.fetch(outputNames[1], outputScores);
        inferenceInterface.fetch(outputNames[2], outputClasses);
        inferenceInterface.fetch(outputNames[3], outputNumDetections);
        Trace.endSection();

        // Find the best detections.
        final PriorityQueue<Recognition> pq =
                new PriorityQueue<Recognition>(
                        1,
                        new Comparator<Recognition>() {
                            @Override
                            public int compare(final Recognition lhs, final Recognition rhs) {
                                // Intentionally reversed to put high confidence at the head of the queue.
                                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                            }
                        });

        // Scale them back to the input size.
        for (int i = 0; i < outputScores.length; ++i) {
            final RectF detection =
                    new RectF(
                            outputLocations[4 * i + 1] * inputSize,
                            outputLocations[4 * i] * inputSize,
                            outputLocations[4 * i + 3] * inputSize,
                            outputLocations[4 * i + 2] * inputSize);
            pq.add(
                    new Recognition("" + i, labels.get((int) outputClasses[i]), outputScores[i], detection));
        }

        final ArrayList<Recognition> recognitions = new ArrayList<Recognition>();
        for (int i = 0; i < Math.min(pq.size(), MAX_RESULTS); ++i) {
            Recognition r = pq.poll();
            if(r.getTitle().contains("person")) {
                recognitions.add(r);
                LOGGER.e("added %s", r.getTitle());
            } else
                LOGGER.e("ignoring %s", r.getTitle());
        }
        Trace.endSection(); // "recognizeImage"
        return recognitions;
    }

    @Override
    public void enableStatLogging(final boolean logStats) {
        this.logStats = logStats;
    }

    @Override
    public String getStatString() {
        return inferenceInterface.getStatString();
    }

    @Override
    public void close() {
        inferenceInterface.close();
    }
}
