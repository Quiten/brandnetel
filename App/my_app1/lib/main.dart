import 'dart:async';
import 'dart:io';
import 'package:image_picker/image_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter/widgets.dart';

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

  Future<void> _pickImage(ImageSource source) async {
    final XFile? picture = await ImagePicker().pickImage(source : source, maxHeight: 400.0, maxWidth: 300.0);
    if (picture != null){
      setState(() {
        _imageFile = picture;
        _image = File(picture.path);
      });
    }
  }

  Future<void> _calculate() async {
    return showDialog<void>(
        context: context,
        builder: (BuildContext context){
          return AlertDialog(
            title: Text("INFO"),
            content: (_imageFile != null)
            ? Text(_imageFile!.path)
            : Text("There's no image"),
            actions: [
              TextButton(
                child: Text("Ok"),
                onPressed: () {
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
                  Text(
                    _imageFile!.path,
                    textAlign: TextAlign.center,
                  ),
                ]),
          )
          : const Center(
              child: Text("Select an image")
              )
        ),
      );
  }
}
