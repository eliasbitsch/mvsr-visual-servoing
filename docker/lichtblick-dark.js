// Dark mode as the default for the lab GUI; a scheme chosen in Lichtblick's settings is kept.
try {
  const key = 'studio.app-configuration.colorScheme';
  if (!localStorage.getItem(key)) localStorage.setItem(key, JSON.stringify('dark'));
} catch (e) {}
