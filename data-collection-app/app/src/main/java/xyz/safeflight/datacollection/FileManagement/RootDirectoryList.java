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

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.os.Environment;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.ListView;
import android.widget.Toast;

import xyz.safeflight.datacollection.R;
import xyz.safeflight.datacollection.Utils;

import java.io.File;

public class RootDirectoryList extends Activity {
    private ListView listView;
    private String[] theNamesOfFiles;
    private String[] filenames;
    private final String TAG = "RootDirectoryList";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_sensor_list);

        listView = ((ListView) findViewById(R.id.listView));

        // Get main directory
        File dir = Utils.getDocumentsDirectory(this,null);

        File[] filelist = dir.listFiles();

        // If no files finish activity and show toast
        if (filelist.length==0){
            Toast.makeText(getApplicationContext(), "There are no recorded flights!", Toast.LENGTH_SHORT).show();
            finish();
        }

        theNamesOfFiles = new String[filelist.length];
        filenames = new String[filelist.length];

        for (int i = 0; i < theNamesOfFiles.length; i++) {
            theNamesOfFiles[i] = Utils.getPrettyDirname(filelist[i].getName());
            filenames[i] = filelist[i].getName();
        }


        ArrayAdapter<String> adapter = new ArrayAdapter<String>(this, android.R.layout.simple_list_item_1, theNamesOfFiles);

        listView.setAdapter(adapter);

        listView.setOnItemClickListener(new AdapterView.OnItemClickListener() {

            public void onItemClick(AdapterView adapterView, View view, int position, long id) {
                if (isDirectory(position)) {
                    Log.e(TAG, "Opening directory: "+ filenames[position]);
                    // Show files inside directory
                    Intent intent = new Intent(view.getContext(), FileList.class);
                    String message = filenames[position];
                    intent.putExtra("filename", message);
                    startActivity(intent);
                }
            }
        });

    }

    private boolean isDirectory(int position){
        File dir = new File(Utils.getDocumentsDirectory(this,null),filenames[position]);
        return dir.isDirectory();
    }


}

