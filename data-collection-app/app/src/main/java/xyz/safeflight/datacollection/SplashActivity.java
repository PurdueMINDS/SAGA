/*
 *  Copyright 2018 Jianfei Gao, Leonardo Teixeira, Bruno Ribeiro.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

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

