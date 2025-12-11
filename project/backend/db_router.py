"""
Database router to direct authentication models to Oracle database
and all other models to SQLite.
"""


class AuthenticationRouter:
    """
    A router to control database operations for authentication models.

    Routes all authentication app models to the 'oracle' database,
    while all other apps use the default SQLite database.
    """

    auth_app_label = 'authentication'

    def db_for_read(self, model, **hints):
        """
        Route read operations for authentication models to Oracle.
        """
        if model._meta.app_label == self.auth_app_label:
            return 'default'  # Temporarily using SQLite
        return 'default'

    def db_for_write(self, model, **hints):
        """
        Route write operations for authentication models to Oracle.
        """
        if model._meta.app_label == self.auth_app_label:
            return 'default'  # Temporarily using SQLite
        return 'default'

    def allow_relation(self, obj1, obj2, **hints):
        """
        Allow relations if both models are in the same database.
        """
        if obj1._meta.app_label == self.auth_app_label and \
           obj2._meta.app_label == self.auth_app_label:
            return True
        elif obj1._meta.app_label != self.auth_app_label and \
             obj2._meta.app_label != self.auth_app_label:
            return True
        return False

    def allow_migrate(self, db, app_label, model_name=None, **hints):
        """
        Ensure authentication models only migrate to Oracle,
        and other models only migrate to SQLite.
        """
        if app_label == self.auth_app_label:
            return db == 'oracle'
        return db == 'default'
