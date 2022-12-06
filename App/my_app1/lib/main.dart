import 'dart:async';
import 'dart:io';
import 'package:image_picker/image_picker.dart';
import 'package:tflite/tflite.dart';
import 'package:meta/meta.dart';
import 'package:flutter/material.dart';
import 'package:flutter/widgets.dart';
// import 'package:tflite_flutter/tflite_flutter.dart';

void main() async {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Brandnetel App',
      theme: ThemeData(brightness: Brightness.dark),
      home: ImageCapture(),
    );
  }
}

class ImageCapture extends StatefulWidget {
  createState() => _ImageCaptureState();
}

class _ImageCaptureState extends State<ImageCapture> {
  XFile? _imageFile;
  File? _image;

  List _result = List.empty();
  String _confidence = ""; 
  String _label = "";

  Future<void> _pickImage(ImageSource source) async {
    var picture = await ImagePicker().pickImage(source : source, maxHeight: 400.0, maxWidth: 300.0);
    
    if (picture != null){
      setState(() {
        _imageFile = picture;
        _image = File(picture.path);
        _applyModel(_image!);
      });
    }
  }

  _loadModel () async {
    var result = await Tflite.loadModel(
      model: "assets/model.tflite", 
      labels: "assets/label.txt");
  }

  _applyModel (File file) async {
    var res = await Tflite.runModelOnImage(
      path: file.path,
      numResults: 2,
      threshold: 0.5,
      imageMean: 127.5,
      imageStd: 127.5, 
      asynch: true        // defaults to true
    );

    setState(() {
      print("Label and Confidence");
      _result = res!;
      print("$_result");
      String str = _result[0]["label"];
      _label = str;
      _confidence = (_result != null )
        ? (_result[0]['confidence']*100.0).toString().substring(0, 2) + "%"
        : "";
      print("Label: $_label and Confidence: $_confidence");
    });
  }

  Future<void> _calculate() async {
    return showDialog<void>(
        context: context,
        builder: (BuildContext context){
          return AlertDialog(
            title: Text("INFO"),
            content: (_imageFile != null)
            // ? Text("Name: $_label \nConfidence: $_confidence")
            ? Text(_imageFile!.path, textAlign: TextAlign.center,)
            : Text("There's no image"),
            actions: [
              TextButton(
                child: Text("Ok"),
                onPressed: () {
                  _applyModel(_image!);
                  Navigator.of(context).pop();
                }, 
                )
            ]
          );
        },
        barrierDismissible: false,
      );

  }

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  @override 
  Widget build(BuildContext context){
    return Scaffold( 
      bottomNavigationBar: BottomAppBar(
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceAround,
          children: <Widget>[
            IconButton(
              icon: Icon(Icons.photo_camera),
              color: Colors.red,
              onPressed: () => _pickImage(ImageSource.camera),
              ),
            IconButton(
              icon: Icon(Icons.photo_library),
              color: Colors.blue,
              onPressed: () => _pickImage(ImageSource.gallery),
              ),
            IconButton(
              icon: Icon(Icons.screen_search_desktop_outlined),
              color: Colors.green,
              onPressed: () => _calculate(), 
              ),
          ],
        ),
      ),
      body: Container(
        child: (_image != null)
          ? Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.spaceAround,
                children: <Widget>[
                  Image.file(_image!),
                  Text("Name: $_label \nConfidence: $_confidence", textAlign: TextAlign.center,)
                ]),
          )
          : const Center(
              child: Text("Select an image")
              )
        ),
      );
  }
}
