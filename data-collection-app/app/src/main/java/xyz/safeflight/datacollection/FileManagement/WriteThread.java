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

package xyz.safeflight.datacollection.FileManagement;

import android.content.Context;
import android.os.Environment;
import android.util.Log;

import xyz.safeflight.datacollection.DataCollection.DataObject;
import xyz.safeflight.datacollection.Utils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;

/**
 * Created by migue on 9/13/2016.
 */
public class WriteThread implements Runnable {
    private ArrayList<DataObject> data;
    //private String filename = Constants.FILENAME;
    private Context context;
    private final String TAG = "WriteThread";
    private int lastIndex = 0;
    private FileSingleton fs;

    public WriteThread(Context context, ArrayList<DataObject> data, int lastIndex){
        this.data = data;
        this.context = context;
        this.lastIndex = lastIndex;
        Log.e(TAG,"Constructor list size is: "+String.valueOf(data.size()));
        fs = FileSingleton.getInstance();

    }

    @Override
    public void run() {
        int len = data.size();
        // Check if media is mounted
        if (isExternalStorageWritable()) {

            // Check if directory exists
            File directory = Utils.getDocumentsDirectory(this.context, fs.getDirectory());
            File outputFile = new File(directory, fs.getFilename());
            Log.e(TAG,"WRITING IN: "+outputFile.getAbsolutePath());

            try {
                FileOutputStream fileOutputStream = new FileOutputStream(outputFile,true);
                OutputStreamWriter outputStreamWriter = new OutputStreamWriter(fileOutputStream);
                for (int i = 0; i < len; i++) {
                    String converted = data.get(i).toString();
                    outputStreamWriter.write(converted);
                }
                outputStreamWriter.close();
            } catch (IOException e) {
                Log.e(TAG, "File write failed: " + e.toString());
            }
        }
    }

    /*private String convertData(DataObject data){
        lastIndex += 1;
        String towrite = String.valueOf(lastIndex) + "\t\t" + data.toString();

        return towrite;
    }*/


    /* Checks if external storage is available for read and write */
    public boolean isExternalStorageWritable() {
        String state = Environment.getExternalStorageState();
        if (Environment.MEDIA_MOUNTED.equals(state)) {
            return true;
        }
        return false;
    }


}
