import 'package:flutter/material.dart';

class StatusScreen extends StatelessWidget {
  final String fatigue;

  const StatusScreen({super.key, required this.fatigue});

  @override
  Widget build(BuildContext context) {
    Color color = Colors.grey;
    if (fatigue == "WARNING_HAPTIC") {
      color = Colors.red;
    } else if (fatigue == "SAFE") {
      color = Colors.green;
    }

    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Text(
            "Fatigue Status",
            style: Theme.of(context).textTheme.headlineMedium,
          ),
          const SizedBox(height: 20),
          Text(
            fatigue,
            style: Theme.of(context).textTheme.displayMedium?.copyWith(color: color),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }
}
