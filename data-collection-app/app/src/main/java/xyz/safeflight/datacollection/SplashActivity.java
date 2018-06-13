package xyz.safeflight.datacollection;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Typeface;
import android.os.Bundle;
import android.util.Log;
import android.widget.ProgressBar;
import android.widget.TextView;

/**
 * Created by migue on 10/5/2016.
 */
    public class SplashActivity extends Activity {
        private int splashTime = 3000;
        private Thread thread;
        private ProgressBar mSpinner;
        private String TAG = "SPLASH";

        @Override
        protected void onCreate(Bundle savedInstanceState) {
            // TODO Auto-generated method stub
            super.onCreate(savedInstanceState);
            Log.e(TAG, "SPLASHHHHHHH");

            setContentView(R.layout.activity_splash);


            mSpinner = (ProgressBar) findViewById(R.id.Splash_ProgressBar);
            mSpinner.setIndeterminate(true);
            thread = new Thread(runable);
            thread.start();
        }
        public Runnable runable = new Runnable() {
            public void run() {
                try {
                    Thread.sleep(splashTime);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                try {
                    startActivity(new Intent(SplashActivity.this,MainActivity.class));
                    finish();
                } catch (Exception e) {
                    // TODO: handle exception
                }
            }
        };
}

