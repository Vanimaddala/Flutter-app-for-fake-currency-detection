import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';

void main() {
  runApp(CurrencyVerificationApp());
}

class CurrencyVerificationApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Currency Verification',
      theme: ThemeData(primarySwatch: Colors.green),
      home: CurrencyVerificationPage(),
    );
  }
}

class CurrencyVerificationPage extends StatefulWidget {
  @override
  _CurrencyVerificationPageState createState() =>
      _CurrencyVerificationPageState();
}

class _CurrencyVerificationPageState extends State<CurrencyVerificationPage> {
  File? _image; // Selected image file
  final ImagePicker _picker = ImagePicker();
  bool _isLoading = false; // Loading state
  String? _result; // Verification result
  Map<String, dynamic>? _features; // Features (variance, skewness, etc.)

  Future<void> _pickImage() async {
    final XFile? pickedFile =
        await _picker.pickImage(source: ImageSource.gallery);
    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
        _result = null;
        _features = null;
      });
    }
  }

  Future<void> _verifyCurrency() async {
    if (_image == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Please select an image to verify.')),
      );
      return;
    }

    setState(() {
      _isLoading = true;
    });

    try {
      final request = http.MultipartRequest(
        'POST',
        Uri.parse('http://127.0.0.1:5000/'),
      );
      request.files.add(await http.MultipartFile.fromPath(
        'my_uploaded_file',
        _image!.path,
      ));

      final response = await request.send();

      if (response.statusCode == 200) {
        final responseData = await response.stream.bytesToString();
        final decodedData = jsonDecode(responseData);

        setState(() {
          _result = decodedData['result'];
          _features = decodedData;
        });
      } else {
        setState(() {
          _result = 'Failed to verify the currency.';
        });
      }
    } catch (e) {
      setState(() {
        _result = 'Error: $e';
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Currency Verification'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            GestureDetector(
              onTap: _pickImage,
              child: Container(
                height: 200,
                width: double.infinity,
                color: Colors.grey[200],
                child: _image == null
                    ? Center(child: Text('Tap to select an image'))
                    : Image.file(_image!, fit: BoxFit.cover),
              ),
            ),
            SizedBox(height: 16),
            ElevatedButton(
              onPressed: _verifyCurrency,
              child: Text('Verify Currency'),
            ),
            SizedBox(height: 16),
            if (_isLoading)
              CircularProgressIndicator()
            else if (_result != null)
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Result: $_result',
                      style:
                          TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                    ),
                    SizedBox(height: 8),
                    if (_features != null)
                      Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text('Variance: ${_features!['variance']}'),
                          Text('Skewness: ${_features!['skewness']}'),
                          Text('Kurtosis: ${_features!['kurtosis']}'),
                          Text('Entropy: ${_features!['entropy']}'),
                        ],
                      ),
                  ],
                ),
              ),
          ],
        ),
      ),
    );
  }
}
