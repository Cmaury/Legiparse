# models.py

from sqlalchemy import (
    Column, Integer, String, Boolean, Text, ForeignKey, Date, DateTime, JSON, Table
)
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from database import engine

Base = declarative_base()

# Association tables
agenda_item_legislation = Table(
    'agenda_item_legislation', Base.metadata,
    Column('agendaitemid', Integer, ForeignKey('agendaitems.agendaitemid'), primary_key=True),
    Column('legislationid', Integer, ForeignKey('legislation.legislationid'), primary_key=True)
)

# Models
class GoverningBody(Base):
    __tablename__ = 'governingbodies'
    governingbodyid = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    contactinfo = Column(JSON)
    city = Column(String(100))
    county = Column(String(100))
    state = Column(String(100))

    meeting_types = relationship('MeetingType', back_populates='governing_body')

class MeetingType(Base):
    __tablename__ = 'meetingtypes'
    meetingtypeid = Column(Integer, primary_key=True)
    governingbodyid = Column(Integer, ForeignKey('governingbodies.governingbodyid'))
    name = Column(String(255), nullable=False)
    description = Column(Text)

    governing_body = relationship('GoverningBody', back_populates='meeting_types')
    meetings = relationship('Meeting', back_populates='meeting_type')

class Meeting(Base):
    __tablename__ = 'meetings'
    meetingid = Column(Integer, primary_key=True)
    meetingtypeid = Column(Integer, ForeignKey('meetingtypes.meetingtypeid'))
    name = Column(String(255))
    agendastatus = Column(String(50))
    minutesstatus = Column(String(50))
    datetime = Column(DateTime, nullable=False)  # Ensure this field is DateTime
    location = Column(String(255))
    publishedagendaurl = Column(Text)
    publishedminutesurl = Column(Text)
    videourl = Column(Text)
    meetingurl = Column(Text, nullable=True)  # New Field: Main Meeting URL
    transcript = Column(Text, nullable=False, default='none')  # If transcripts are lengthy
    processed = Column(Boolean, default=False)

    meeting_type = relationship('MeetingType', back_populates='meetings')
    agendas = relationship('Agenda', back_populates='meeting')
    media = relationship('Media', back_populates='meeting')  # Ensure relationship is defined

class Agenda(Base):
    __tablename__ = 'agendas'
    agendaid = Column(Integer, primary_key=True)
    meetingid = Column(Integer, ForeignKey('meetings.meetingid'))
    title = Column(String(255))
    description = Column(Text)

    meeting = relationship('Meeting', back_populates='agendas')
    agenda_items = relationship('AgendaItem', back_populates='agenda')

class AgendaItem(Base):
    __tablename__ = 'agendaitems'
    agendaitemid = Column(Integer, primary_key=True)
    agendaid = Column(Integer, ForeignKey('agendas.agendaid'))
    order = Column(Integer)
    title = Column(Text)
    description = Column(Text)
    readabledescription = Column(Text)
    is_interesting = Column(Boolean, nullable=False, default=False)

    agenda = relationship('Agenda', back_populates='agenda_items')
    legislations = relationship('Legislation', secondary=agenda_item_legislation, back_populates='agenda_items')

class Legislation(Base):
    __tablename__ = 'legislation'
    legislationid = Column(Integer, primary_key=True)
    filenumber = Column(String(50))
    version = Column(Integer)
    type = Column(String(100))
    status = Column(String(100))
    filecreateddate = Column(Date)
    incontrol = Column(String(255))
    onagendadate = Column(Date)
    finalactiondate = Column(Date)
    enactmentdate = Column(Date)
    enactmentnumber = Column(String(50))
    effectivedate = Column(Date)
    title = Column(Text)
    sponsors = Column(JSON)
    indexes = Column(Text)
    text = Column(Text)
    readabletext = Column(Text)
    is_interesting = Column(Boolean, nullable=False, default=False)

    agenda_items = relationship('AgendaItem', secondary=agenda_item_legislation, back_populates='legislations')
    history = relationship('LegislationHistory', back_populates='legislation')
    attachments = relationship('Attachment', back_populates='legislation')

class LegislationHistory(Base):
    __tablename__ = 'legislationhistory'
    historyid = Column(Integer, primary_key=True)
    legislationid = Column(Integer, ForeignKey('legislation.legislationid'))
    actiondate = Column(Date)
    version = Column(Integer)
    action = Column(String(255))
    result = Column(String(50))
    actiondetailsurl = Column(Text)
    meetingdetailsurl = Column(Text)
    videourl = Column(Text)

    legislation = relationship('Legislation', back_populates='history')

class Attachment(Base):
    __tablename__ = 'attachments'
    attachmentid = Column(Integer, primary_key=True)
    legislationid = Column(Integer, ForeignKey('legislation.legislationid'))
    filename = Column(String(255))
    fileurl = Column(Text)

    legislation = relationship('Legislation', back_populates='attachments')

class Media(Base):
    __tablename__ = 'media'
    mediaid = Column(Integer, primary_key=True)
    meetingid = Column(Integer, ForeignKey('meetings.meetingid'))
    mediatype = Column(String(50))
    url = Column(Text)
    description = Column(Text)

    meeting = relationship('Meeting', back_populates='media')  # Ensure back_populates is set