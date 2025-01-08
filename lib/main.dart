import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:typed_data';
import 'dart:io' as io;
import 'package:image_picker/image_picker.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Currency Detection',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: CurrencyDetectionScreen(),
    );
  }
}

class CurrencyDetectionScreen extends StatefulWidget {
  @override
  _CurrencyDetectionScreenState createState() => _CurrencyDetectionScreenState();
}

class _CurrencyDetectionScreenState extends State<CurrencyDetectionScreen> {
  String _result = "Upload an image for detection";
  bool _isUploading = false;

  Future<void> _uploadImage(io.File imageFile) async {
    setState(() {
      _isUploading = true;
    });

    final String url = 'http://127.0.0.1:5000/'; // Flask server URL

    try {
      var request = http.MultipartRequest('POST', Uri.parse(url));
      final stream = imageFile.openRead();
      final String filename = imageFile.path.split('/').last;
      final multipartFile = http.MultipartFile(
        'my_uploaded_file',
        stream,
        await imageFile.length(),
        filename: filename,
      );
      request.files.add(multipartFile);

      var response = await request.send();
      var responseBody = await response.stream.bytesToString();

      if (response.statusCode == 200) {
        final data = json.decode(responseBody);
        setState(() {
          _result = "Result: ${data['result']}\nVariance: ${data['variance']}\nSkew: ${data['skew']}\nKurtosis: ${data['kurtosis']}\nEntropy: ${data['entropy']}";
        });
      } else {
        setState(() {
          _result = "Failed to upload image. Status code: ${response.statusCode}";
        });
      }
    } catch (e) {
      setState(() {
        _result = "Error: $e";
      });
    } finally {
      setState(() {
        _isUploading = false;
      });
    }
  }

  Future<void> _pickImage() async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: ImageSource.gallery);

    if (pickedFile != null) {
      print("Picked image path: ${pickedFile.path}");
      io.File imageFile = io.File(pickedFile.path);
      _uploadImage(imageFile);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Currency Detection"),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            _isUploading
                ? CircularProgressIndicator()
                : ElevatedButton(
                    onPressed: _pickImage,
                    child: Text("Pick Image for Detection"),
                  ),
            SizedBox(height: 20),
            Text(_result, textAlign: TextAlign.center),
          ],
        ),
      ),
    );
  }
}
