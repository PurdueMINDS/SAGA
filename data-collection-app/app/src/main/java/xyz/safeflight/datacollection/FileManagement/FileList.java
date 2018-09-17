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
import android.content.Intent;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.ListView;

import xyz.safeflight.datacollection.R;
import xyz.safeflight.datacollection.Utils;

import java.io.File;

/**
 * Created by migue on 10/14/2016.
 */
public class FileList extends Activity {
    private ListView listView;
    private String[] theNamesOfFiles;
    private String[] filenames;
    private String directory;
    private final String TAG = "RootDirectoryList";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_sensor_list);

        listView = ((ListView) findViewById(R.id.listView));

        Intent intent = getIntent();
        directory = intent.getStringExtra("filename");

        // Get main directory
        File dir = Utils.getDocumentsDirectory(this,directory);

        File[] filelist = dir.listFiles();
        theNamesOfFiles = new String[filelist.length];
        filenames = new String[filelist.length];

        for (int i = 0; i < theNamesOfFiles.length; i++) {
            theNamesOfFiles[i] = Utils.getPrettyFilename(filelist[i].getName());
            filenames[i] = filelist[i].getName();
        }

        ArrayAdapter<String> adapter = new ArrayAdapter<String>(this, android.R.layout.simple_list_item_1, theNamesOfFiles);

        listView.setAdapter(adapter);

        listView.setOnItemClickListener(new AdapterView.OnItemClickListener() {

            public void onItemClick(AdapterView adapterView, View view, int position, long id) {
                Log.e(TAG, "Opening directory: "+ filenames[position]);
                // Show files inside directory
                Intent intent = new Intent(view.getContext(), FileActivity.class);
                String message = filenames[position];
                intent.putExtra("filename", directory + "/" + message);
                startActivity(intent);
            }
        });

    }


}

