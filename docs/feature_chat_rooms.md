# Chat rooms

This document describes an addition to the TalkWithMe app.
The goal is to add chat rooms and allow personas to be assigned/removed to/from chat rooms.
This allows large lists of configured personas to be broken up into logical groupings
and allows the user to easily switch between those groupings.

## Current state

The left panel shows only the "Who should answer?" chooser and a flat list of all configured personas.
The main chat panel shows a singular chat view, which can be cleared via the "New Chat" control in the top right.

## Desired state

Add a "Chat room" dropdown above the "Who should answer?" chooser.
This dropdown should default to the chat room named "default".
All configured personas are always assigned to the "default" chat room.
The "default" chat room cannot be deleted. Personas cannot be unassigned from it.

### Defining chat rooms

Add a new "Chat rooms" control in the top right, in between the "Personas" and "Settings" controls.
This brings up a list of all configured chat rooms (not including the "default" chat room, which
cannot be edited or deleted). Users should be able to remove any chat room from this list, with a
confirmation dialog to confirm the change. Users should also be able to add a new chat room, by
specifying a name for it (20 chars max), which must be unique (case-insensitive). Users cannot
create a chat room called "default" (or any case variation of "default").

When a chat room is created, it is initially empty (no personas are assigned).

Closing the chat room editor updates the contents of the chat room dropdown in the left panel.
If the channel that was previously selected no longer exists, the dropdown should revert to
the "default" chat room.

Chat rooms cannot be renamed once created - user must delete and create a new one.

### Assigning personas to a chat room

Selecting a chat room from the chat room dropdown in the left panel should update the personas
list to show the personas assigned to that room, if any. Selecing the "default" chat room
should always show all configured personas.

Add a new "Add persona" control underneath the "Personas" label in the left menu but above the
list of personas. Selecting this control should bring up a list of all configured personas. The user can select one
and it will be added to the persona list for the current chat room. If possible, multi-select should
be an option.

If the user changes to the "default" chat room, hide the "Add persona" control and simply show all
configured personas.

Personas can be assigned to any number of chat rooms simultaneously.

### Removing personas from a chat room

If the current chat room is not the "default" chat room, show a small red "x" control to the right of each
persona in the persona list in the left panel. Selecting this control will unassign that persona from the current
chat room. The persona is not deleted - simply unassigned from that room.

If the user changes to the "default" chat room, hide this "remove" control.

## Switching chat rooms

Changing the current chat room blanks out the main chat panel, exactly as if the user had selected
the "New Chat" control. Persistent chats is a future feature; don't worry about it for now.

## Renaming a persona

Personas can be renamed in the persona editor! When this occurs, the code should update all chat rooms
to which this persona was assigned so that they have the updated name.

## Deleting a persona

Personas can be deleted in the persona editor! All chat rooms should be updated to remove the
deleted persona, if they were assigned to that chat room.

## Chat room persistence

Chat rooms should be defined in a new yaml configuration file called `chatrooms.yaml`, alongside the
existing `settings.yaml` and `personas.yaml`. The configuration is simple: just a list of all defined
chat rooms (except "default", which always exists implicitly), and for each chat room, a list of assigned personas.

It is not an error condition if this file does not exist on startup - the code can gracefully fall
back to the implicit "default" chat room, which shows all available personas. The chat room editor
can create/overwrite this file as changes are made.

## Chatting in an empty room

If no personas are configured, or if the user switches to a chat room with no personas assigned,
then sending any chat message (either by typing or via microphone input) should NOT follow the usual
chat flow, but should instead show an error saying "No one is here."

