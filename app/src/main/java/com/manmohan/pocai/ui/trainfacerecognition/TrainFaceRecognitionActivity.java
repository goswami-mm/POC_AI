package com.manmohan.pocai.ui.trainfacerecognition;

import android.content.ClipData;
import android.content.ContentResolver;
import android.content.Intent;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.provider.MediaStore;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.FileProvider;

import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.Toast;

import com.manmohan.pocai.R;
import com.manmohan.pocai.aiengine.classifiers.TrainPersonDetectionClassifier;
import com.manmohan.pocai.aiengine.env.FileUtils;

import java.io.File;
import java.util.ArrayList;

import static com.manmohan.pocai.aiengine.env.ImageUtils.getBitmapFromUri;

public class TrainFaceRecognitionActivity extends AppCompatActivity {

    private static final int IMAGE_FILE_CAPTURE = 12;
    private Button takePicBt;
    private Button galleryPicBt;
    private Button trainBt;
    private EditText personNameEt;
    private ImageView imgCapture;
    private static final int IMAGE_CAPTURE_CODE = 1;
    private ProgressBar progressBar;
    private Uri mImageUri;
    private TrainPersonDetectionClassifier trainPersonDetectionClassifier;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_train_face_recognition);

        takePicBt = findViewById(R.id.take_pic_bt);
        galleryPicBt = findViewById(R.id.gallery_pic_bt);
        progressBar = findViewById(R.id.progress_bar);
        trainBt = findViewById(R.id.train_model_bt);
        personNameEt = findViewById(R.id.person_name_et);
        imgCapture = findViewById(R.id.image);

        takePicBt.setOnClickListener(v -> {
            openCamera();
        });

        findViewById(R.id.other_bt).setOnClickListener(v -> {
            startActivity(new Intent(this, TrainFaceRecognition.class));
        });

        galleryPicBt.setOnClickListener(v -> {
            performFileSearch();
        });
        trainBt.setOnClickListener(v->{
            if(personNameEt.getText().toString().trim().isEmpty()){
                Toast.makeText(this, "Fill person name", Toast.LENGTH_LONG).show();
            } else if(mImageUri == null){
                Toast.makeText(this, "No image found", Toast.LENGTH_LONG).show();
            } else {
                progressBar.setVisibility(View.VISIBLE);
                new Thread(() -> {
                    try {
                        trainPersonDetectionClassifier.updateData(
                                personNameEt.getText().toString().trim(),
                                getContentResolver(), mImageUri);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                    runOnUiThread(() -> {
                        mImageUri = null;
                        imgCapture.setImageResource(R.drawable.ic_launcher_background);
                        progressBar.setVisibility(View.GONE);
                    });
                }).start();
            }
        });

        new Thread(this::init).start();
    }

    private void openCamera() {
        Intent intent = new Intent("android.media.action.IMAGE_CAPTURE");
        File photo;
        try {
            // place where to store camera taken picture
            photo = this.createTemporaryFile("picture", ".jpg");
//            photo.delete();
        } catch(Exception e) {
            Toast.makeText(this, "Please check SD card! Image shot is impossible!", Toast.LENGTH_LONG);
            return;
        }
        mImageUri = FileProvider.getUriForFile(this,
                "com.manmohan.pocai.provider", //(use your app signature + ".provider" )
                photo);
        intent.putExtra(MediaStore.EXTRA_OUTPUT, mImageUri);
        //start camera intent
        startActivityForResult(intent, IMAGE_CAPTURE_CODE);
    }
    public void setImage(ImageView imageView)
    {
//        this.getContentResolver().notifyChange(mImageUri, null);
//        ContentResolver cr = this.getContentResolver();
        Bitmap bitmap;
        try {
            bitmap = getBitmapFromUri(getContentResolver(), mImageUri);
            imageView.setImageBitmap(bitmap);
        } catch (Exception e) {
            Toast.makeText(this, "Failed to load", Toast.LENGTH_SHORT).show();
        }
    }

    private File createTemporaryFile(String part, String ext) throws Exception
    {
        File dir = new File(FileUtils.ROOT);
        dir = new File(dir.getAbsolutePath()+"/temp/");
        if(!dir.exists())
        {
            dir.mkdirs();
        }
        return new File(dir, part+""+System.currentTimeMillis()+ ext);
    }

    void init() {
        runOnUiThread(()-> progressBar.setVisibility(View.VISIBLE));
        File dir = new File(FileUtils.ROOT);

        if (!dir.isDirectory()) {
            if (dir.exists()) dir.delete();
            dir.mkdirs();

            AssetManager mgr = getAssets();
            FileUtils.copyAsset(mgr, FileUtils.DATA_FILE);
            FileUtils.copyAsset(mgr, FileUtils.MODEL_FILE);
            FileUtils.copyAsset(mgr, FileUtils.LABEL_FILE);
        }

        try {
            trainPersonDetectionClassifier = TrainPersonDetectionClassifier.create(getAssets());
        } catch (Exception e) {
            finish();
        }
        runOnUiThread(()-> progressBar.setVisibility(View.GONE));
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode == IMAGE_CAPTURE_CODE) {
            if (resultCode == RESULT_OK) {
                setImage(imgCapture);
            } else if (resultCode == RESULT_CANCELED) {
                Toast.makeText(this, "Cancelled", Toast.LENGTH_LONG).show();
            }

        } else if (requestCode == IMAGE_FILE_CAPTURE){
            if (resultCode == RESULT_OK) {
                mImageUri = data.getData();
                setImage(imgCapture);
            }
        }
    }

    public void performFileSearch() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true);
        intent.setType("image/*");

        startActivityForResult(intent, IMAGE_FILE_CAPTURE);
    }
}
