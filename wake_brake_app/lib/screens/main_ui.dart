import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../providers/fatigue_provider.dart';
import 'camera_screen.dart';
import 'status_screen.dart';
import 'monitor_screen.dart';
import 'settings_screen.dart';

// ignore_for_file: library_private_types_in_public_api, use_key_in_widget_constructors

class MainUI extends ConsumerStatefulWidget {
  const MainUI({super.key});

  @override
  ConsumerState<MainUI> createState() => _MainUIState();
}

class _MainUIState extends ConsumerState<MainUI> {
  int _selectedTab = 0;

  @override
  Widget build(BuildContext context) {
    final fatigueState = ref.watch(fatigueProvider);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Wake&Brake Companion'),
        actions: [
          IconButton(
            icon: const Icon(Icons.warning, color: Colors.red),
            onPressed: () {
              ref.read(fatigueProvider.notifier).panicReset();
            },
            tooltip: 'Panic Reset',
          )
        ],
      ),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _selectedTab,
        onDestinationSelected: (index) {
          setState(() {
            _selectedTab = index;
          });
        },
        destinations: const [
          NavigationDestination(icon: Text("📷"), label: "Camera"),
          NavigationDestination(icon: Text("📊"), label: "Status"),
          NavigationDestination(icon: Text("📡"), label: "Monitor"),
          NavigationDestination(icon: Text("⚙️"), label: "Config"),
        ],
      ),
      body: IndexedStack(
        index: _selectedTab,
        children: [
          const CameraScreen(),
          StatusScreen(fatigue: fatigueState.fatigue),
          const MonitorScreen(),
          const SettingsScreen(),
        ],
      ),
    );
  }
}
